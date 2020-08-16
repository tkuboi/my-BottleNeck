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
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.regularizers import l2
#from tensorflow.keras.backend import l2_normalize
from tensorflow.math import l2_normalize
from tensorflow import math as math_ops
from tensorflow import dtypes

import functools
from functools import partial

from utils import compose

InceptionConv2D = partial(Conv2D, padding='same', activation='relu')
InceptionMaxPooling2D = partial(MaxPooling2D, padding='same')
InceptionConcatenate = partial(concatenate, axis=3)

@functools.wraps(Conv2D)
def inception_conv2d(*args, **kwargs):
    return InceptionConv2D(*args, **kwargs)

def inception_max_pooling2d(*args, **kwargs):
    return InceptionMaxPooling2D(*args, **kwargs)

def inception_l2_normalize(**kwargs):
    normalize_kwargs = {'axis': 1}
    normalize_kwargs.update(kwargs)
    return partial(l2_normalize, **normalize_kwargs) 

def inception_concatenate(*args, **kwargs):
    return InceptionConcatenate(*args, **kwargs)

def base_layer1():
    return compose(
            inception_conv2d(64, kernel_size=(7, 7), strides=2, name='conv2d_base_0'),
            inception_max_pooling2d(pool_size=(3, 3), strides=2, name='max_pooling2d_base_0'),
            inception_l2_normalize(name='base_l2_normalize_0'),
            )

def base_layer2():
    return compose(
            inception_conv2d(64, kernel_size=(1, 1), strides=1, name='conv2d_base_1'),
            inception_conv2d(192, kernel_size=(3, 3), strides=1, name='conv2d_base_2'),
            inception_l2_normalize(name='base_l2_normalize_1'),
            inception_max_pooling2d(pool_size=(3, 3), strides=2, name='max_pooling2d_base_1')
            )

def inception_s1_unit(filters, strides=1):
    return inception_conv2d(filters, kernel_size=(1, 1), strides=strides)

def inception_s3_unit(reduction_filters, filters, strides=1):
    return compose(
            inception_conv2d(reduction_filters, kernel_size=(1, 1), strides=1),
            inception_conv2d(filters, kernel_size=(3, 3), strides=strides)
            )

def inception_s5_unit(reduction_filters, filters, strides=1):
    return compose(
            inception_conv2d(reduction_filters, kernel_size=(1, 1), strides=1),
            inception_conv2d(filters, kernel_size=(5, 5), strides=strides)
            )

def inception_pooling_unit(pool_size=(3, 3), strides=(1, 1),
                           filters=0, kernel_size=(1, 1), conv_strides=1):
    if filters:
        return compose(
                inception_max_pooling2d(pool_size=pool_size, strides=strides),
                inception_conv2d(filters, kernel_size=kernel_size, strides=conv_strides)
                )
    return inception_max_pooling2d(pool_size=pool_size, strides=strides)

def inception_normalizing_unit(filters, kernel_size=(1, 1), strides=1):
    return compose(
            inception_l2_normalize(),
            inception_conv2d(filters, kernel_size=kernel_size, strides=strides)
            )

def inception_block_3a(x):
    x1 = inception_s1_unit(64)(x)
    x2 = inception_s3_unit(96, 128)(x)
    x3 = inception_s5_unit(16, 32)(x)
    x4 = inception_pooling_unit(filters=32)(x)
    return concatenate([x1, x2, x3, x4], axis=3)

def inception_block_3b(x):
    x1 = inception_s1_unit(64)(x)
    x2 = inception_s3_unit(96, 128)(x)
    x3 = inception_s5_unit(32, 64)(x)
    x4 = inception_normalizing_unit(filters=64)(x)
    return concatenate([x1, x2, x3, x4], axis=3)

def inception_block_3c(x):
    x2 = inception_s3_unit(128, 256)(x)
    x3 = inception_s5_unit(32, 64)(x)
    x4 = inception_pooling_unit(filters=0)(x)
    return concatenate([x2, x3, x4], axis=3)

def inception_block_4a(x):
    x1 = inception_s1_unit(256)(x)
    x2 = inception_s3_unit(96, 192)(x)
    x3 = inception_s5_unit(32, 64)(x)
    x4 = inception_normalizing_unit(filters=128)(x)
    return concatenate([x1, x2, x3, x4], axis=3)

def inception_block_4b(x):
    x1 = inception_s1_unit(224)(x)
    x2 = inception_s3_unit(112, 224)(x)
    x3 = inception_s5_unit(32, 64)(x)
    x4 = inception_normalizing_unit(filters=128)(x)
    return concatenate([x1, x2, x3, x4], axis=3)

def inception_block_4c(x):
    x1 = inception_s1_unit(192)(x)
    x2 = inception_s3_unit(128, 256)(x)
    x3 = inception_s5_unit(32, 64)(x)
    x4 = inception_normalizing_unit(filters=128)(x)
    return concatenate([x1, x2, x3, x4], axis=3)

def inception_block_4d(x):
    x1 = inception_s1_unit(160)(x)
    x2 = inception_s3_unit(144, 288)(x)
    x3 = inception_s5_unit(32, 64)(x)
    x4 = inception_normalizing_unit(filters=128)(x)
    return concatenate([x1, x2, x3, x4], axis=3)

def inception_block_4e(x):
    x2 = inception_s3_unit(160, 256, strides=2)(x)
    x3 = inception_s5_unit(64, 128, strides=2)(x)
    x4 = inception_pooling_unit(filters=0, strides=2)(x)
    return concatenate([x2, x3, x4], axis=3)

def inception_block_5a(x):
    x1 = inception_s1_unit(384)(x)
    x2 = inception_s3_unit(192, 384)(x)
    x3 = inception_s5_unit(48, 128)(x)
    x4 = inception_normalizing_unit(filters=128)(x)
    return concatenate([x1, x2, x3, x4], axis=3)

def inception_block_5b(x):
    x1 = inception_s1_unit(384)(x)
    x2 = inception_s3_unit(192, 384)(x)
    x3 = inception_s5_unit(48, 128)(x)
    x4 = inception_pooling_unit(filters=128)(x)
    return concatenate([x1, x2, x3, x4], axis=3)

def inception_model(x, embedding_size=128):
    x = inception_block_3a(x)
    x = inception_block_3b(x)
    x = inception_block_3c(x)
    x = inception_block_4a(x)
    x = inception_block_4b(x)
    x = inception_block_4c(x)
    x = inception_block_4d(x)
    x = inception_block_4e(x)
    x = inception_block_5a(x)
    x = inception_block_5b(x)
    x = AveragePooling2D(pool_size=(1, 1), strides=1)(x)
    x = Flatten()(x)
    x = Dense(embedding_size, activation=None)(x)
    x = inception_l2_normalize()(x)
    return x

def base_model(image_input_shape, embedding_size):
    input_image = Input(shape=image_input_shape)
    x = base_layer1()(input_image)
    x = base_layer2()(x)
    x = Conv2D(192, kernel_size=(19, 25), strides=25, activation='relu', name='conv2d_head_0')(x)
    x = Flatten()(x)
    x = Dense(embedding_size, activation=None)(x)
    x = inception_l2_normalize()(x)

    model = Model(inputs=input_image, outputs=x)
    return model

def whole_model(image_input_shape, embedding_size):
    input_image = Input(shape=image_input_shape)
    x = base_layer1()(input_image)
    x = base_layer2()(x)
    x = inception_block_3a(x)
    x = inception_block_3b(x)
    x = inception_block_3c(x)
    x = inception_block_4a(x)
    x = inception_block_4b(x)
    x = inception_block_4c(x)
    x = inception_block_4d(x)
    x = inception_block_4e(x)
    x = inception_block_5a(x)
    x = inception_block_5b(x)
    #x = AveragePooling2D(pool_size=(1, 1), strides=1)(x)
    x = Flatten()(x)
    x = Dense(embedding_size, activation=None)(x)
    x = inception_l2_normalize()(x)
    model = Model(inputs=input_image, outputs=x)
    return model
