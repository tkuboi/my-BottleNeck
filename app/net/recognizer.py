"""
Author: Toshi
"""
from PIL import Image
import numpy as np
import time
import json
import sys
import os 
from os import path

#if __package__ is None:
sys.path.append(path.dirname( path.abspath(__file__) ) )
from siamese_model import SiameseModel
NET_DIR = path.dirname(path.abspath(__file__))
MODEL_FILE = '%s/models/model.h5' % (NET_DIR)
CLASS_IMAGES_DIR = '%s/np_samples' % (NET_DIR)
BOTTLE_DIR = '%s/bottles' % (NET_DIR)

class Recognizer:
    def __init__(self):
        self.model_file = MODEL_FILE 
        self.app = None
        self.model = SiameseModel()
        #self.graph = SiameseModel.get_default_graph() 
        self.class_images, self.classes =\
            self._load_classes(CLASS_IMAGES_DIR)

    def _load_classes(self, class_images_dir):
        class_images = []
        classes = []
        sub_dir_list = os.listdir(class_images_dir)
        for item in sub_dir_list:
            tokens = item.split('.')
            if len(tokens) <= 1:
                continue
            filepath = os.path.join(class_images_dir, item)
            if not os.path.isfile(filepath):
                print(filepath, "is not a file")
                continue
            if tokens[-1] == 'npy':
                class_images = np.load(filepath)
            elif tokens[-1] == 'json':
                with open(filepath, "r") as content:
                    classes = json.load(content)
        return class_images, classes

    def init_app(self, app):
        self.app = app

    def load_model(self):
        #with self.app.app_context():
        self.model.load_model(self.model_file)

    def predict(self, filepath):
        filepath = '%s/%s' % (BOTTLE_DIR, filepath)
        test_images, names = self.model.preprocess_images(filepath)
        image = test_images[0]
        name = names[0]
        label = []
        score = []
        for c, sample in enumerate(self.class_images):
            image, sample = image.reshape((1, -1)), sample.reshape((1, -1))
            score.append(self.model.predict([image, sample])[0])
            label.append(c)

        index = np.argmax(score)
        label_ = self.classes[label[index]]
        print('IMAGE {} is {} with confidence of {}'.format(
            name, label_, score[index][0]))

        return label_, score[index][0]

