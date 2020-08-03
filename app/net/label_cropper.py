"""A cropper for wine labels using the yolo detector.
This is the step 1 out of 3 for the preparation of sample wine embeddings.

Author: Toshi
"""
import argparse
import sys
import os
import traceback

import PIL
from PIL import Image

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

from utils import to_np
from yolo import yolo
from yolo import yolo_eval
from yolo_utils import resize_boundingboxes, get_classes, get_anchors, preprocess_image

class LabelCropper:
    image_extensions = ['.JPG', '.jpeg', '.jpg', '.png']

    def __init__(self, model_path, weights_path,
                 classes_path=None,
                 anchors_path=None,
                 is_proportional_anchor=True,
                 input_dimension=320):
        self.input_dimension = input_dimension
        self.classes = get_classes(classes_path)
        self.anchors = get_anchors(
                        anchors_path,
                        is_proportional=is_proportional_anchor,
                        image_wh=self.input_dimension)
        yolo_body = load_model(model_path)
        yolo_body.load_weights(weights_path)
        self.detector = yolo(yolo_body, self.anchors, len(self.classes))
        self.max_boxes = 1
        self.score_threshold = .3
        self.iou_threshold = .9

    def crop_label(self, file_path):
        image = Image.open(file_path)
        if image.size[0] > image.size[1]:
            image = image.rotate(-90)
        image_input = preprocess_image(image, [self.input_dimension, self.input_dimension])
        yolo_outputs = self.detector(image_input)
        out_boxes, out_scores, out_classes = yolo_eval(
            yolo_outputs,
            [image.size[1], image.size[0]],
            max_boxes=self.max_boxes,
            score_threshold=self.score_threshold,
            iou_threshold=self.iou_threshold)

        if len(out_boxes) == 0:
            return None
        bounding_boxes = resize_boundingboxes(out_boxes, image.size)
        bounding_box = bounding_boxes[0]
        cropped = image.crop((bounding_box[1], bounding_box[0], bounding_box[3], bounding_box[2]))
        return cropped

    def read_directory(self, in_dir, out_dir):
        count = 0
        for item in os.listdir(in_dir):
            item_path = os.path.join(in_dir, item)
            if os.path.isdir(item_path):
                count += self.read_directory(item_path, out_dir)
            else:
                name, ext = os.path.splitext(item)
                if 'label' in name or ext not in self.image_extensions:
                    continue
                try:
                    label = self.crop_label(item_path)
                    if label:
                        out_dir_path = LabelCropper.check_outdir(
                                os.path.join(out_dir, os.path.basename(in_dir)))
                        label.save(os.path.join(out_dir_path, name + '_label.jpg'))
                        count += 1
                except:
                    traceback.print_exc(file=sys.stdout)
        return count

    @staticmethod
    def check_outdir(dir_path):
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        return dir_path

    def crop_labels(self, in_dir, out_dir):
        return self.read_directory(in_dir, out_dir)

def main(args):
    model_path = os.path.expanduser(args.model_path)
    weights_path = os.path.expanduser(args.weights_path)
    classes_path = os.path.expanduser(args.classes_path)
    anchors_path = os.path.expanduser(args.anchors_path)
    input_path = os.path.expanduser(args.in_dir_path)
    output_path = os.path.expanduser(args.out_dir_path)
    image_wh = args.dimension_input_image
    is_proportional_anchor = args.proportional_anchor

    cropper = LabelCropper(model_path, weights_path,
                           classes_path=classes_path,
                           anchors_path=anchors_path,
                           is_proportional_anchor=is_proportional_anchor,
                           input_dimension=image_wh)
    count = cropper.crop_labels(input_path, output_path)
    print(count)

if __name__ == '__main__':
    argparser = argparse.ArgumentParser(
        description='Detect labels in images with yolo and crop them.')
    argparser.add_argument(
        '-m',
        '--model_path',
        help='path to HDF5 file containing a yolo model and weights',
        default='model_data/yolo_body.h5')

    argparser.add_argument(
        '-w',
        '--weights_path',
        help='path to previously trained yolo weights',
        default='model_data/yolo_weights.h5')

    argparser.add_argument(
        '-a',
        '--anchors_path',
        help='path to anchors file, defaults to yolo_anchors.txt',
        default='model_data/wine_anchors.txt')

    argparser.add_argument(
        '-c',
        '--classes_path',
        help='path to classes file, defaults to pascal_classes.txt',
        default='model_data/wine_classes.txt')

    argparser.add_argument(
        '-d',
        '--dimension_input_image',
        type=int,
        help='input image shape for yolo',
        default=320)

    argparser.add_argument(
        '-i',
        '--in_dir_path',
        help='path to a directory containing images',
        default='bottles')

    argparser.add_argument(
        '-o',
        '--out_dir_path',
        help='path to a directory where labels will be saved',
        default='labels')

    argparser.add_argument(
        '-p',
        '--proportional_anchor',
        help='the anchors are proportional number between 0 and 1.0.',
        action='store_true',
        default=False)

    main(argparser.parse_args())
