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
from tensorflow.keras.optimizers import Adam, Adagrad, RMSprop

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

def get_anchors(anchors_path, is_proportional=False, image_wh=608):
    '''loads the anchors from a file'''
    if os.path.isfile(anchors_path):
        with open(anchors_path) as f:
            anchors = f.readline()
        if is_proportional:
            anchors = [float(x) * image_wh / 32 for x in anchors.split(',')]
        else:
            anchors = [float(x) for x in anchors.split(',')]
        return np.array(anchors).reshape(-1, 2)
    else:
        Warning("Could not open anchors file, using default.")
        return YOLO_ANCHORS

def load_training_data(data_path):
    image_data = np.load(os.path.join(data_path, 'x.npy'))
    boxes = np.load(os.path.join(data_path, 'y.npy'))
    return image_data, boxes

def get_detector_mask(boxes, anchors, image_wh=608):
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
            preprocess_true_boxes(box, anchors, [image_wh, image_wh])

    return np.array(detectors_mask), np.array(matching_true_boxes)

def create_model(model_path, anchors, class_names, **kw):
    if 'weights_path' in kw:
        weights_path = kw['weights_path']
        load_saved_weights = True if weights_path else False
    else:
        load_saved_weights = False
        weights_path = None
    if 'freeze_body' in kw:
        freeze_body = kw['freeze_body']
    else:
        freeze_body = True
    #not currently used
    if 'input_shape' in kw:
        input_shape = kw['input_shape']
    else:
        input_shape = (608, 608, 3)
    # Create model body.
    print("CREATING TOPLESS WEIGHTS FILE")
    model_body = load_model(model_path)
    topless_yolo = Model(model_body.inputs, model_body.layers[-2].output)

    if freeze_body:
        for layer in topless_yolo.layers:
            layer.trainable = False

    final_layer = Conv2D(
            len(anchors)*(5+len(class_names)), (1, 1),
            activation='linear', name='conv2d_final'
        )(topless_yolo.output)

    model = Model(topless_yolo.inputs, final_layer)
    if load_saved_weights:
        model.load_weights(weights_path)

    return model

def train(model, class_names, anchors, image_data, boxes, detectors_mask, matching_true_boxes, **kw):
    if 'learning_rate' in kw:
        learning_rate = kw['learning_rate']
    else:
        learning_rate = 0.001
    if 'validation_split' in kw:
        validation_split = kw['validation_split']
    else:
        validation_split = 0.1
    if 'num_epochs' in kw:
        num_epochs = kw['num_epochs']
    else:
        num_epochs = 30
    if 'batch_size' in kw:
        batch_size = kw['batch_size']
    else:
        batch_size = 8
    if 'decay_rate' in kw:
        decay_rate = kw['decay_rate']
    else:
        decay_rate = 1.0 
    if 'decay_steps' in kw:
        decay_steps = kw['decay_steps']
    else:
        decay_steps = 100

    num_classes = len(class_names)
    loss_funcs = [yolo_loss2(anchors, num_classes, mask, box)
                  for mask, box in zip(detectors_mask, matching_true_boxes)]
    steps_per_epoch = image_data.shape[0] * (1.0 - validation_split) // batch_size
    lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(
                          learning_rate,
                          decay_steps=steps_per_epoch*decay_steps,
                          decay_rate=decay_rate,
                          staircase=False)
    model.compile(optimizer=Adam(lr_schedule), loss=loss_funcs)
    logging = TensorBoard()
    checkpoint_path = "model_data/yolo_v2_%d_weights_ep{epoch:02d}_BS%d_best.h5" % (
                        image_data.shape[1],
                        batch_size)
    checkpoint = ModelCheckpoint(checkpoint_path,
                                 monitor='val_loss',
                                 save_weights_only=True, save_best_only=True)
    early_stopping = EarlyStopping(
        monitor='val_loss', min_delta=0, patience=15, verbose=1, mode='auto')

    #print(image_data.shape)
    #print(boxes.shape)
    history = model.fit(
                image_data, boxes,
                validation_split=validation_split,
                batch_size=batch_size,
                epochs=num_epochs,
                callbacks=[logging, checkpoint, early_stopping])
   
    model.save('model_data/yolo_body_v2_%d.h5' % (image_data.shape[1]))
    model.save_weights('model_data/yolo_v2_%d_weights_ep%02d_BS%d_vloss%s.h5' % (
        image_data.shape[1], num_epochs, batch_size,
        ('%0.5f' % history.history['val_loss'][0]).replace('.', '_')))

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
    model_path = os.path.expanduser(args.model_path)
    data_path = os.path.expanduser(args.data_path)
    classes_path = os.path.expanduser(args.classes_path)
    anchors_path = os.path.expanduser(args.anchors_path)
    weights_path = os.path.expanduser(args.weights_path)
    test_path = os.path.expanduser(args.test_path)
    output_path = os.path.expanduser(args.output_path)
    num_epochs = args.epochs
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    decay_rate = args.decay_rate
    decay_steps = args.decay_steps
    validation_split = args.validation_split
    image_wh = args.input_dimension
    is_gpu = args.gpu
    is_proportional_anchor = args.proportional_anchor

    if not is_gpu:
        tf.config.set_visible_devices([], 'GPU')

    if not os.path.exists(output_path):
        print('Creating output path {}'.format(output_path))
        os.mkdir(output_path)

    class_names = get_classes(classes_path)

    anchors = get_anchors(anchors_path, is_proportional_anchor, image_wh)
    model = create_model(
            model_path, anchors, class_names,
            weights_path=weights_path, input_shape=(image_wh, image_wh, 3))
    model.summary()
    #print(images[3].size)
    #print(images[4].size)
    image_data, boxes = load_training_data(data_path) 
    #print(boxes)
    detector_mask, matching_true_boxes = get_detector_mask(boxes, anchors, image_wh)
    #print(matching_true_boxes)
    train(
            model, class_names, anchors, image_data,
            boxes, detector_mask, matching_true_boxes,
            num_epochs=num_epochs, batch_size=batch_size,
            learning_rate=learning_rate,
            decay_rate=decay_rate,
            decay_steps=decay_steps,
            validation_split=validation_split)
    evaluate(model, class_names, anchors, test_path, output_path)

if __name__ == '__main__':
    argparser = argparse.ArgumentParser(
    description='Train YOLO_v2 model to overfit on a single image.')

    argparser.add_argument(
        '-m',
        '--model_path',
        help='path to HDF5 file containing a yolo model and weights',
        default='model_data/yolo.h5')

    argparser.add_argument(
        '-d',
        '--data_path',
        help='path to a directory containing numpy arrays (X and Y) of training dataset',
        default='model_data/')

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
        '-w',
        '--weights_path',
        help='path to previously trained yolo weights',
        default='')

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

    argparser.add_argument(
        '-l',
        '--learning_rate',
        help='the learning rate',
        type=float,
        default=0.001)

    argparser.add_argument(
        '-dr',
        '--decay_rate',
        help='the decay rate',
        type=float,
        default=0.5)

    argparser.add_argument(
        '-ds',
        '--decay_steps',
        help='the decay steps',
        type=int,
        default=100)

    argparser.add_argument(
        '-v',
        '--validation_split',
        help='the validation split',
        type=float,
        default=0.1)

    argparser.add_argument(
        '-g',
        '--gpu',
        help='use gpu',
        action='store_true',
        default=False)

    argparser.add_argument(
        '-p',
        '--proportional_anchor',
        help='the anchors are proportional number between 0 and 1.0.',
        action='store_true',
        default=False)

    argparser.add_argument(
        '-i',
        '--input_dimension',
        type=int,
        help='input image shape',
        default=608)

    args = argparser.parse_args()
    main(args)
