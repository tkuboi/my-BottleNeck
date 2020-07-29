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
from yolo import preprocess_true_boxes

def get_classes(classes_path):
    '''loads the classes'''
    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names

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
                images.append(np.asarray(image))
                image.close()
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
    im = Image.fromarray(images[0])
    orig_size = np.array([im.width, im.height])
    orig_size = np.expand_dims(orig_size, axis=0)

    # Image preprocessing.
    processed_images = [
        Image.fromarray(i).resize(input_image_size, PIL.Image.BICUBIC) for i in images]
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

def main(args):
    data_path = os.path.expanduser(args.data_path)
    classes_path = os.path.expanduser(args.classes_path)
    image_wh = args.image_wh

    class_names = get_classes(classes_path)

    images, boxes = read_directory(data_path)
    image_data, boxes = create_training_data(
            images, boxes, input_image_size=(image_wh, image_wh))
    del images
    np.save(os.path.join(data_path, 'x.npy'), image_data)
    np.save(os.path.join(data_path, 'y.npy'), boxes)

if __name__ == '__main__':
    argparser = argparse.ArgumentParser(
    description='Generate YOLO training data from a proprietrary corrdinates file.')

    argparser.add_argument(
        '-d',
        '--data_path',
        help='path to HDF5 file containing pascal voc dataset',
        default='tmp_labels2')

    argparser.add_argument(
        '-c',
        '--classes_path',
        help='path to classes file, defaults to pascal_classes.txt',
        default='model_data/coco_classes.txt')

    argparser.add_argument(
        '-w',
        '--image_wh',
        type=int,
        help='input image width and height',
        default=608)

    args = argparser.parse_args()
    main(args)
