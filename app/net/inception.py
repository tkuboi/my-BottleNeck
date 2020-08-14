"""Contains Inception net blocks
Dependecies:
    tensorflow v2.2
"""
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import concatenate 
from tensorflow.keras.layers import Dense 
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import Input 
from tensorflow.keras.layers import Flatten 
from tensorflow.keras.regularizers import l2
from tensorflow.keras.backend import l2_normalize
from tensorflow import math as math_ops
from tensorflow import dtypes

import functools
from functools import partial

from utils import compose

InceptionConv2D = partial(Conv2D, padding='same', activation='relu')
InceptionMaxPooling2D = partial(MaxPooling2D, padding='same')

@functools.wraps(Conv2D)
def inception_conv2d(*args, **kwargs):
    return InceptionConv2D(*args, **kwargs)

def inception_max_pooling2d(*args, **kwargs):
    return InceptionMaxPooling2D(*args, **kwargs)

def inception_l2_normalize():
    return partial(l2_normalize, axis=1) 

def base_layer1():
    return compose(
            inception_conv2d(64, kernel_size=(7, 7), strides=2),
            inception_max_pooling2d(pool_size=(3, 3), strides=2),
            inception_l2_normalize(),
            )

def base_layer2():
    return compose(
            inception_conv2d(64, kernel_size=(1, 1), strides=1),
            inception_conv2d(192, kernel_size=(3, 3), strides=1),
            inception_l2_normalize(),
            inception_max_pooling2d(pool_size=(3, 3), strides=2)
            )

def inception_unit(s1, s3r, s3=(18, 1), s5r=32, s5=(64, 1), pooling=None, l2=0):
    pass 

def inception_block(out):
    pass

def base_model(image_input_shape, embedding_size):
    input_image = Input(shape=image_input_shape)
    x = base_layer1()(input_image)
    x = base_layer2()(x)
    x = Conv2D(192, kernel_size=(19, 25), strides=25, activation='relu')(x)
    x = Flatten()(x)
    x = Dense(embedding_size, activation=None)(x)
    x = inception_l2_normalize()(x)

    model = Model(inputs=input_image, outputs=x)
    return model 
