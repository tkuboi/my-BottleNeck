"""A recognizer for wines

Author: Toshi
"""
import pickle
from collections import OrderedDict
import sys
import os
import traceback
import configparser

import PIL
from PIL import Image

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from yolo import yolo
from yolo import yolo_eval
from yolo_utils import draw_boxes
from retrain_yolo import get_classes, get_anchors
from utils import to_np

class WineRecognizer:
    base_dir = os.path.dirname(os.path.abspath(__file__))

    def __init__(self, config_path):
        self.read_config(config_path)
        self.classes = get_classes(self.classes_path)
        self.anchors = get_anchors(
                        self.anchors_path,
                        is_proportional=self.is_proportional_anchor,
                        image_wh=self.input_dimension)
        self.app = None
        self.detector = None
        self.recognizer = None
        self.wines_dict = pickle.load(open(self.wines_dict_path, 'rb'))
        self.wine_embeddings = np.load(self.sample_embeddings_path)
        self.wine_ids = np.load(self.sample_ids_path)

    def read_config(self, config_path):
        config = configparser.ConfigParser()
        config.read(config_path)
        if 'Yolo' in config:
            if 'model_path' in config['Yolo']:
                self.yolo_model_path = os.path.join(self.base_dir, config['Yolo']['model_path']) 
            if 'weights_path' in config['Yolo']:
                self.yolo_weights_path = os.path.join(self.base_dir, config['Yolo']['weights_path']) 
            if 'classes_path' in config['Yolo']:
                self.classes_path = os.path.join(self.base_dir, config['Yolo']['classes_path'])
            if 'anchors_path' in config['Yolo']:
                self.anchors_path = os.path.join(self.base_dir, config['Yolo']['anchors_path']) 
            if 'out_path' in config['Yolo']:
                self.out_path = os.path.join(self.base_dir, config['Yolo']['out_path']) 
            if 'is_proportional_anchor' in config['Yolo']:
                self.is_proportional_anchor = config['Yolo'].getboolean('is_proportional_anchor') 
            if 'input_dimension' in config['Yolo']:
                self.input_dimension = config['Yolo'].getint('input_dimension') 
            if 'max_boxes' in config['Yolo']:
                self.max_boxes = config['Yolo'].getint('max_boxes') 
            if 'score_threshold' in config['Yolo']:
                self.score_threshold = config['Yolo'].getfloat('score_threshold') 
            if 'iou_threshold' in config['Yolo']:
                self.iou_threshold = config['Yolo'].getfloat('iou_threshold') 
        if 'Facenet' in config:
            if 'model_path' in config['Facenet']:
                self.facenet_model_path = os.path.join(self.base_dir, config['Facenet']['model_path'])
            if 'weights_path' in config['Facenet']:
                self.facenet_weights_path = os.path.join(self.base_dir, config['Facenet']['weights_path'])
            if 'sample_embeddings_path' in config['Facenet']:
                self.sample_embeddings_path = os.path.join(self.base_dir, config['Facenet']['sample_embeddings_path'])
            if 'sample_ids_path' in config['Facenet']:
                self.sample_ids_path = os.path.join(self.base_dir, config['Facenet']['sample_ids_path'])
            if 'wines_dict_path' in config['Facenet']:
                self.wines_dict_path = os.path.join(self.base_dir, config['Facenet']['wines_dict_path'])
            if 'image_width' in config['Facenet']:
                self.label_width = config['Facenet'].getint('image_width') 
            if 'image_height' in config['Facenet']:
                self.label_height = config['Facenet'].getint('image_height') 
            if 'k_neighbors' in config['Facenet']:
                self.k_neighbors = config['Facenet'].getint('k_neighbors') 

    def init_app(self, app):
        self.app = app
        self.load_models()

    def load_models(self):
        model_body = load_model(self.yolo_model_path)
        model_body.load_weights(self.yolo_weights_path)
        self.detector = yolo(model_body, self.anchors, len(self.classes))
        self.recognizer = load_model(self.facenet_model_path)
        self.recognizer.load_weights(self.facenet_weights_path)

    def predict(self, file_path):
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

        bounding_boxes = adjust_boundingbox(out_boxes, image.size)
        if len(bounding_boxes) == 0:
            return None, None
        # Plot image with predicted boxes.
        image_with_boxes = draw_boxes(image.copy(), bounding_boxes, out_classes,
                                      self.classes, out_scores, False)
        image_file = os.path.basename(file_path)
        image_with_boxes.save(os.path.join(self.out_path, image_file), quality=90)
        bounding_box = bounding_boxes[0]
        cropped = image.crop((bounding_box[1], bounding_box[0], bounding_box[3], bounding_box[2]))
        cropped.save(os.path.join(self.out_path, 'cropped.JPG'))
        wines = self.recognize(cropped)
        return wines, [int(bounding_box[1]),
                       int(bounding_box[0]),
                       int(bounding_box[3]),
                       int(bounding_box[2])]

    def recognize(self, image):
        #label_data = np.array(image.resize((self.label_width, self.label_height)))
        dim = (self.label_width, self.label_height)
        label_data = np.array([to_np(image.resize(dim), dim)])
        label_data = label_data.astype('float32')
        label_data /= 255.
        label_embedding = self.recognizer.predict(label_data)
        neighbors = get_neighbors(
                self.wine_embeddings, self.wine_ids, label_embedding, self.k_neighbors)
        #print(neighbors)
        results = map_wines(neighbors[0], self.wines_dict)
        return results

def preprocess_image(image, model_image_shape):
    resized_image = image.resize(
            tuple(reversed(model_image_shape)), Image.BICUBIC)
    image_data = np.array(resized_image, dtype='float32')
    image_data /= 255.
    image_input = np.expand_dims(image_data, 0)  # Add batch dimension.
    return image_input

def adjust_boundingbox(out_boxes, image_size):
    boxes = [] 
    for out_box in out_boxes:
        top, left, bottom, right = out_box
        top = max(0, np.floor(top).astype('int32'))
        left = max(0, np.floor(left).astype('int32'))
        bottom = min(image_size[1], np.floor(bottom).astype('int32'))
        right = min(image_size[0], np.floor(right).astype('int32'))
        print(top, left, bottom, right)
        width = (right - left)
        height = (bottom - top)
        c_x = left + width // 2
        c_y = top + height // 2
        if width // 3  >= height // 4:
            width = int(width)
            height = width * 4 // 3
        else:
            height = int(height)
            width = height * 3 // 4
        box = (c_y - height // 2, c_x - width // 2, c_y + height // 2, c_x + width // 2)
        boxes.append(box)
    return boxes

def get_dists(embeddings, test):
    dists = [None] * len(embeddings)
    for i, embedding in enumerate(embeddings):
        dists[i] = np.linalg.norm(embedding - test)
    return dists

def get_neighbors(embeddings, y_train, tests, k):
    results = []
    for i, test in enumerate(tests):
        dists = get_dists(embeddings, test)
        zipped = zip(dists, y_train)
        zipped = list(zipped)
        neighbors = sorted(zipped, key=lambda x : x[0])
        results.append(k_nearest(neighbors, k))
    return results

def k_nearest(neighbors, k):
    n = 0
    top_k = OrderedDict()
    for neighbor in neighbors:
        if neighbor[1] not in top_k:
            top_k[neighbor[1]] = neighbor[0]
            n += 1
        if n == k:
            break
    return [(v, int(k)) for k, v in top_k.items()]

def map_wines(neighbors, wines):
    results = []
    for k, v in neighbors:
        results.append((v, wines[v], str(k)))
    return results 

if __name__ == '__main__':
    recognizer = WineRecognizer('recognizer.cfg')
    recognizer.load_models()
    results, box = recognizer.predict('images/bottle3.JPG')
    for result in results:
        print(result)

