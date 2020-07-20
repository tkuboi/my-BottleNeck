import argparse
import io
import os
import sys
import traceback
import colorsys
import imghdr
import random

import h5py
import matplotlib.pyplot as plt
import numpy as np
import PIL
from PIL import Image
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Input, Lambda, Conv2D
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import EarlyStopping 

from yolo import (preprocess_true_boxes, yolo_body, yolo,
                                     yolo_eval, yolo_head, yolo_loss2)
from yolo_utils import draw_boxes

YOLO_ANCHORS = np.array(
    ((0.57273, 0.677385), (1.87446, 2.06253), (3.33843, 5.47434),
     (7.88282, 3.52778), (9.77052, 9.16828)))

def get_classes(classes_path):
    '''loads the classes'''
    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names

def get_anchors(anchors_path):
    '''loads the anchors from a file'''
    if os.path.isfile(anchors_path):
        with open(anchors_path) as f:
            anchors = f.readline()
            anchors = [float(x) for x in anchors.split(',')]
            return np.array(anchors).reshape(-1, 2)
    else:
        Warning("Could not open anchors file, using default.")
        return YOLO_ANCHORS

def read_images(file_path):
    """read images from a file
    Args:
        file_path (str): the path to a file
    Returns:
        list: a list of Image objects
        list: a list of bounding boxes
    """
    images = []
    boxes = []
    dir_path = os.path.dirname(file_path) 
    with open(file_path) as f:
        for line in f:
            try:
                tokens = line.split("\t")
                image_name = os.path.join(dir_path, tokens[0])
                image = Image.open(image_name)
                if image.size[0] > image.size[1]:
                    image = image.rotate(-90, expand=1)
                images.append(image)
                box = [39] + list(map(int, tokens[1:]))
                #print(box)
                boxes.append(box)
            except:
                traceback.print_exc(file=sys.stdout)
    return images, boxes 

def read_directory(dir_path):
    images = []
    boxes = []
    for item in os.listdir(dir_path):
        path = os.path.join(dir_path, item)
        if os.path.isdir(path):
            ims, bxs = read_directory(path)
            images += ims
            boxes += bxs
        elif "coordinates" in item:
            ims, bxs = read_images(path)
            images += ims
            boxes += bxs
            
    return images, boxes

def create_training_data(images, boxes=None, **kw):
    if "input_image_size" in kw:
        input_image_size = kw["input_image_size"]
    else:
        input_image_size = (608, 608)
    orig_size = np.array([images[0].width, images[0].height])
    orig_size = np.expand_dims(orig_size, axis=0)

    # Image preprocessing.
    processed_images = [i.resize(input_image_size, PIL.Image.BICUBIC) for i in images]
    processed_images = [np.array(image, dtype=np.float) for image in processed_images]
    processed_images = [image/255. for image in processed_images]

    if boxes is not None:
        # Box preprocessing.
        # Original boxes stored as 1D list of class, x_min, y_min, x_max, y_max.
        boxes = [np.array(box).reshape((-1, 5)) for box in boxes]
        # Get extents as y_min, x_min, y_max, x_max, class for comparision with
        # model output.
        boxes_extents = [box[:, [2, 1, 4, 3, 0]] for box in boxes]

        # Get box parameters as x_center, y_center, box_width, box_height, class.
        boxes_xy = [0.5 * (box[:, 3:5] + box[:, 1:3]) for box in boxes]
        boxes_wh = [box[:, 3:5] - box[:, 1:3] for box in boxes]
        boxes_xy = [boxxy / orig_size for boxxy in boxes_xy]
        boxes_wh = [boxwh / orig_size for boxwh in boxes_wh]
        boxes = [
            np.concatenate((boxes_xy[i], boxes_wh[i], box[:, 0:1]), axis=1)
            for i, box in enumerate(boxes)]

        # find the max number of boxes
        max_boxes = 0
        for boxz in boxes:
            if boxz.shape[0] > max_boxes:
                max_boxes = boxz.shape[0]

        # add zero pad for training
        for i, boxz in enumerate(boxes):
            if boxz.shape[0]  < max_boxes:
                zero_padding = np.zeros( (max_boxes-boxz.shape[0], 5), dtype=np.float32)
                boxes[i] = np.vstack((boxz, zero_padding))

        return np.array(processed_images), np.array(boxes)
    else:
        return np.array(processed_images)

def get_detector_mask(boxes, anchors):
    '''
    Precompute detectors_mask and matching_true_boxes for training.
    Detectors mask is 1 for each spatial position in the final conv layer and
    anchor that should be active for the given boxes and 0 otherwise.
    Matching true boxes gives the regression targets for the ground truth box
    that caused a detector to be active or 0 otherwise.
    '''
    detectors_mask = [0 for i in range(len(boxes))]
    matching_true_boxes = [0 for i in range(len(boxes))]
    for i, box in enumerate(boxes):
        detectors_mask[i], matching_true_boxes[i] =\
            preprocess_true_boxes(box, anchors, [608, 608])

    return np.array(detectors_mask), np.array(matching_true_boxes)

def create_model(anchors, class_names, load_pretrained=True, freeze_body=True):
    #detectors_mask_shape = (13, 13, 5, 1)
    #matching_boxes_shape = (13, 13, 5, 5)

    # Create model body.
    print("CREATING TOPLESS WEIGHTS FILE")
    yolo_path = os.path.join('model_data', 'yolo.h5')
    topless_yolo_path = os.path.join('model_data', 'yolo_topless.h5')
    model_body = load_model(yolo_path)
    #model_body.summary()
    topless_yolo = Model(model_body.inputs, model_body.layers[-2].output)
    #topless_yolo.summary()
    if load_pretrained:
        topless_yolo.load_weights(topless_yolo_path)
    topless_yolo.save_weights(topless_yolo_path)

    if freeze_body:
        for layer in topless_yolo.layers:
            layer.trainable = False
    final_layer = Conv2D(
            len(anchors)*(5+len(class_names)), (1, 1),
            activation='linear', name='conv2d_final'
        )(topless_yolo.output)

    model_body = Model(topless_yolo.inputs, final_layer)
    #model = yolo(model_body, anchors, len(class_names))
    return model_body

def train(model, class_names, anchors, image_data, boxes, detectors_mask, matching_true_boxes, **kw):
    if 'validation_split' in kw:
        validation_split = kw['validation_split']
    else:
        validation_split=0.1
    if 'num_epochs' in kw:
        num_epochs = kw['num_epochs']
    else:
        num_epochs = 30
    if 'batch_size' in kw:
        batch_size = kw['batch_size']
    else:
        batch_size = 8

    num_classes = len(class_names)
    loss_funcs = [yolo_loss2(anchors, num_classes, mask, box)
                  for mask, box in zip(detectors_mask, matching_true_boxes)]
    model.compile(optimizer='adam', loss=loss_funcs)
    logging = TensorBoard()
    checkpoint = ModelCheckpoint("yolo_v2_weights_best.h5", monitor='val_loss',
                                 save_weights_only=True, save_best_only=True)
    early_stopping = EarlyStopping(
        monitor='val_loss', min_delta=0, patience=15, verbose=1, mode='auto')

    #print(image_data.shape)
    #print(boxes.shape)
    model.fit(image_data, boxes,
              validation_split=validation_split,
              batch_size=batch_size,
              epochs=num_epochs,
              callbacks=[logging, checkpoint, early_stopping])

    model.save_weights('yolo_v2_weights_ep%02d_BS%d.h5' % (num_epochs, batch_size))

def evaluate(model, class_names, anchors, test_path, output_path):
    # Create output variables for prediction.
    yolo_model = yolo(model, anchors, len(class_names))

    model_image_size = yolo_model.layers[0].input_shape[0][1:3]
    is_fixed_size = model_image_size != (None, None)

    for image_file in os.listdir(test_path):
        try:
            image_type = imghdr.what(os.path.join(test_path, image_file))
            if not image_type:
                continue
        except IsADirectoryError:
            continue

        image = Image.open(os.path.join(test_path, image_file))
        if is_fixed_size:  # TODO: When resizing we can use minibatch input.
            resized_image = image.resize(
                tuple(reversed(model_image_size)), Image.BICUBIC)
            image_data = np.array(resized_image, dtype='float32')
        else:
            # Due to skip connection + max pooling in YOLO_v2, inputs must have
            # width and height as multiples of 32.
            new_image_size = (image.width - (image.width % 32),
                              image.height - (image.height % 32))
            resized_image = image.resize(new_image_size, Image.BICUBIC)
            image_data = np.array(resized_image, dtype='float32')
            #print(image_data.shape)

        image_data /= 255.
        image_input = np.expand_dims(image_data, 0)  # Add batch dimension.

        yolo_outputs = yolo_model(image_input)
        out_boxes, out_scores, out_classes = yolo_eval(
            yolo_outputs,
            [image.size[1], image.size[0]],
            max_boxes=1,
            score_threshold=.3,
            iou_threshold=.9)

        print('Found {} boxes for {}'.format(len(out_boxes), image_file))

        # Plot image with predicted boxes.
        image_with_boxes = draw_boxes(image, out_boxes, out_classes,
                                      class_names, out_scores, False)
        #image_with_boxes = Image.fromarray(image_with_boxes)
        image_with_boxes.save(os.path.join(output_path, image_file), quality=90)

def main(args):
    data_path = os.path.expanduser(args.data_path)
    classes_path = os.path.expanduser(args.classes_path)
    anchors_path = os.path.expanduser(args.anchors_path)
    test_path = os.path.expanduser(args.test_path)
    output_path = os.path.expanduser(args.output_path)
    num_epochs = args.epochs
    batch_size = args.batch_size

    if not os.path.exists(output_path):
        print('Creating output path {}'.format(output_path))
        os.mkdir(output_path)

    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]

    if os.path.isfile(anchors_path):
        with open(anchors_path) as f:
            anchors = f.readline()
            anchors = [float(x) for x in anchors.split(',')]
            anchors = np.array(anchors).reshape(-1, 2)
    else:
        anchors = YOLO_ANCHORS

    model = create_model(anchors, class_names, False)
    model.summary()
    images, boxes = read_directory(data_path)
    #print(images[3].size)
    #print(images[4].size)
    image_data, boxes = create_training_data(images, boxes)
    #print(boxes)
    detector_mask, matching_true_boxes = get_detector_mask(boxes, anchors)
    #print(matching_true_boxes)
    train(
            model, class_names, anchors, image_data,
            boxes, detector_mask, matching_true_boxes,
            num_epochs=num_epochs, batch_size=batch_size)
    evaluate(model, class_names, anchors, test_path, output_path)

if __name__ == '__main__':
    argparser = argparse.ArgumentParser(
    description='Train YOLO_v2 model to overfit on a single image.')

    argparser.add_argument(
        '-d',
        '--data_path',
        help='path to HDF5 file containing pascal voc dataset',
        default='tmp_labels2')

    argparser.add_argument(
        '-a',
        '--anchors_path',
        help='path to anchors file, defaults to yolo_anchors.txt',
        default='model_data/yolo_anchors.txt')

    argparser.add_argument(
        '-c',
        '--classes_path',
        help='path to classes file, defaults to pascal_classes.txt',
        default='model_data/coco_classes.txt')

    argparser.add_argument(
        '-t',
        '--test_path',
        help='path to directory of test images, defaults to images/',
        default='images')

    argparser.add_argument(
        '-o',
        '--output_path',
        help='path to output test images, defaults to images/out',
        default='images/out')

    argparser.add_argument(
        '-e',
        '--epochs',
        help='the number of epochs',
        type=int,
        default=30)

    argparser.add_argument(
        '-b',
        '--batch_size',
        help='the batch size',
        type=int,
        default=8)

    args = argparser.parse_args()
    main(args)
