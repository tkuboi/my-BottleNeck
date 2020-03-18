import time
import os
import numpy as np
import traceback
import tensorflow as tf
from tensorflow.keras import models , optimizers , losses ,activations , callbacks
from tensorflow.keras.layers import *
import tensorflow.keras.backend as K
from PIL import Image
from tensorflow import keras
from image_utils import crop_image

class SiameseModel:
    def __init__(self, dim=128):
        self.dim = dim
        input_shape = ((self.dim**2) * 3,)
        convolution_shape = (self.dim, self.dim, 3)
        kernel_size_1 = (4, 4)
        kernel_size_2 = (3, 3)
        pool_size_1 = (3, 3)
        pool_size_2 = (2, 2)
        strides = 1

        seq_conv_model = [
            Reshape( input_shape=input_shape , target_shape=convolution_shape),
            Conv2D(32, kernel_size=kernel_size_1, strides=strides, activation=tf.nn.leaky_relu),
            Conv2D(32, kernel_size=kernel_size_1, strides=strides, activation=tf.nn.leaky_relu),
            MaxPooling2D(pool_size=pool_size_1, strides=strides),
            Conv2D(64, kernel_size=kernel_size_2, strides=strides, activation=tf.nn.leaky_relu),
            Conv2D(64, kernel_size=kernel_size_2, strides=strides, activation=tf.nn.leaky_relu),
            MaxPooling2D(pool_size=pool_size_2, strides=strides),
            Flatten(),
            Dense(64, activation=activations.sigmoid)
        ]

        seq_model = tf.keras.Sequential(seq_conv_model)
        input_x1 = Input(shape=input_shape)
        input_x2 = Input(shape=input_shape)
        output_x1 = seq_model(input_x1)
        output_x2 = seq_model(input_x2)

        distance_euclid = Lambda(lambda tensors : K.abs(tensors[0] - tensors[1]))([output_x1, output_x2])
        outputs = Dense(1, activation=activations.sigmoid) (distance_euclid)
        self.model = models.Model([input_x1, input_x2], outputs)
        self.model.compile(loss=losses.binary_crossentropy, optimizer=optimizers.Adam(lr=0.0001))

    def fit(self, X, Y, hyperparameters):
        initial_time = time.time()
        self.model.fit(X, Y,
            batch_size=hyperparameters['batch_size'] ,
            epochs=hyperparameters['epochs'],
            callbacks=hyperparameters['callbacks'],
            validation_data=hyperparameters['val_data']
        )
        final_time = time.time()
        eta = (final_time - initial_time)
        time_unit = 'seconds'
        if eta >= 60:
            eta = eta / 60
            time_unit = 'minutes'
        self.model.summary( )
        print('Elapsed time acquired for {} epoch(s) -> {} {}'.format(hyperparameters['epochs'], eta, time_unit))


    def evaluate(self, test_X, test_Y) :
        return self.model.evaluate(test_X, test_Y)

    def predict(self, X):
        predictions = self.model.predict(X)
        return predictions

    def summary(self):
        self.model.summary()

    def save_model(self, file_path):
        self.model.save(file_path)

    def load_model(self, file_path):
        self.model = models.load_model(file_path,
                custom_objects={'leaky_relu': tf.nn.leaky_relu})
        print(file_path)

    def preprocess_images(self, dir_path , flatten=True):
        samples = []
        names = []
        if os.path.isdir(dir_path):
            files = os.listdir(dir_path)
        else:
            files = [dir_path]
            dir_path = ''
        for file_path in files:
            path = os.path.join(dir_path, file_path)
            try:
                image_data = self.process_image(path)
                samples.append(image_data)
                names.append(file_path)
            except:
                print('WARNING : File {} could not be processed.'.format(path))
                print(traceback.format_exc())
        if flatten :
            samples = np.array(samples)
            return samples.reshape((samples.shape[0], self.dim**2 * 3)).astype(np.float32), names
        return np.array(samples), names 

    def process_image(self, path):
        image = Image.open(path)
        w, h = image.size
        cropped = crop_image(image, (0, h - w, w, h))
        resized_image = cropped.resize((self.dim, self.dim))
        arr = []
        for x in range(self.dim):
            sub_array = []
            for y in range(self.dim):
                sub_array.append(resized_image.load()[x, y])
            arr.append(sub_array)
        image_data = np.array(arr)
        image_data = np.array(np.reshape(image_data, (self.dim, self.dim, 3))) / 255
        return image_data

    @staticmethod
    def get_default_graph():
        return tf.Graph()
