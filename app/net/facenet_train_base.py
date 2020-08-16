import tensorflow as tf
import tensorflow_addons as tfa
## dataset
from tensorflow.keras.datasets import mnist

## for Model definition/training
from tensorflow.keras.models import Model, load_model
#from tensorflow.keras.layers import Input, Flatten, Dense, concatenate,  Dropout
from tensorflow.keras.layers import * 
from tensorflow.keras.optimizers import Adam, Adagrad, RMSprop
from tensorflow.keras import regularizers
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.backend import l2_normalize
## required for semi-hard triplet loss:
#from tensorflow.python.ops import tf
#from tensorflow.python.ops import math_ops
from tensorflow import math as math_ops
#from tensorflow.python.framework import dtypes
from tensorflow import dtypes

## for visualizing 
import matplotlib.pyplot as plt
import numpy as np

import os
import sys 
import traceback
import argparse

from sklearn.decomposition import PCA

from PIL import Image

from inception import base_model
from inception import inception_model 
from inception import whole_model 

def train_body(model,
               input_image_shape=(150, 200, 3), embedding_size=128,
               batch_size=32, epochs=30, lr=0.0001, decay_rate=1.0,
               decay_steps=100, x_train = None, y_train=None,
               x_val=None, y_val=None, x_test=None, y_test=None):

    lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(
                      lr,
                      decay_steps=decay_steps,
                      decay_rate=1,
                      staircase=False)
    # train session
    opt = Adam(lr_schedule)  # choose optimiser. RMS is good too!

    # Network training...
    model.compile(loss=tfa.losses.TripletSemiHardLoss(), optimizer=opt)

    checkpoint_path = "facenet_v5_ep{epoch:02d}_BS%d/cp.ckpt" % batch_size
    checkpoint = ModelCheckpoint(
            checkpoint_path, verbose=1, save_best_only=True, save_weights_only=True, period=1)
    callbacks_list = [checkpoint]

    history = model.fit(
            x=x_train,
            y=y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(x_val, y_val),
            callbacks=callbacks_list)

    model.save_weights(checkpoint_path.format(epoch=epochs))
    plot_train_val_loss(history)

    return model

def create_model_body(base_model_path=None, base_weights_path=None, 
                      model_path=None, weights_path=None, freeze=True):
    if model_path:
        model = load_model(model_path)
        if weights_path:
            model.load_weights(weights_path)
        if freeze:
            for layer in model.layers[:10]:
                layer.trainable = False
    else: 
        base_model = load_model(base_model_path)
        base_model.load_weights(base_weights_path)
        topless_base = Model(base_model.inputs, base_model.layers[-8].output)
        if freeze:
            for layer in topless_base.layers:
                layer.trainable = False
        body_output = inception_model(topless_base.output)
        model = Model(topless_base.inputs, body_output)
        model.save('saved_model/facenet_v5_model')
    model.summary()
    plot_model(model, to_file='inception_network.png', show_shapes=True, show_layer_names=True)
    return model

def train_base(train_flag, model_path=None, weights_path=None,
               input_image_shape=(150, 200, 3), embedding_size=128,
               batch_size=32, epochs=30, lr=0.0001, decay_rate=1.0,
               decay_steps=100, x_train = None, y_train=None,
               x_val=None, y_val=None, x_test=None, y_test=None):
    # Network training...
    if train_flag == True:
        if model_path:
            model = load_model(model_path)
        else:
            #model = base_model(input_image_shape, embedding_size)
            model = whole_model(input_image_shape, embedding_size)
        plot_model(model, to_file='base_network.png', show_shapes=True, show_layer_names=True)
        #input_images = Input(shape=input_image_shape, name='input_image') # input layer for images
        #input_labels = Input(shape=(1,), name='input_label')    # input layer for labels
        #embeddings = model([input_images])               # output of network -> embeddings

        model.summary()
        model.save('saved_model/facenet_base_model')
        plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=True)

        lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(
                          0.0001,
                          decay_steps=decay_steps,
                          decay_rate=1,
                          staircase=False)
        # train session
        opt = Adam(lr_schedule)  # choose optimiser. RMS is good too!

        model.compile(loss=tfa.losses.TripletSemiHardLoss(), optimizer=opt)

        if weights_path:
            model.load_weights(weights_path)

        checkpoint_path = "facenet_base_ep{epoch:02d}_BS%d/cp.ckpt" % batch_size
        checkpoint = ModelCheckpoint(
                checkpoint_path, verbose=1, save_best_only=True, save_weights_only=True, period=1)
        callbacks_list = [checkpoint]

        print(x_train.shape)
        print(x_val.shape)
        history = model.fit(
            x=x_train,
            y=y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(x_val, y_val),
            callbacks=callbacks_list)

        model.save_weights(checkpoint_path.format(epoch=epochs))

        plot_train_val_loss(history)

    else:

        #####
        model = load_model(model_path)
        model.load_weights(weights_path)

    return model

def plot_train_val_loss(training_history):
        fig = plt.figure(figsize=(8,8))
        plt.plot(training_history.history['loss'], label='training loss')
        plt.plot(training_history.history['val_loss'], label='validation loss')
        plt.legend()
        plt.title('Train/validation loss')
        plt.show()
        fig.savefig('train_val_loss.png', bbox_inches='tight')
        return

def test_trained_model(before, after, x_test, y_test, epochs):
    testing_embeddings = before
    x_embeddings_before_train = testing_embeddings.predict(x_test)
    # Grabbing the weights from the trained network
    for layer_target, layer_source in zip(testing_embeddings.layers, after.layers):
        weights = layer_source.get_weights()
        layer_target.set_weights(weights)
        del weights

    # Visualizing the effect of embeddings -> using PCA!
    x_embeddings = testing_embeddings.predict(x_test)
    dict_embeddings = {}
    dict_gray = {}
    test_class_labels = np.unique(y_test)

    no_of_components = 2  # for visualization -> PCA.fit_transform()

    step = 1

    pca = PCA(n_components=no_of_components)
    decomposed_embeddings = pca.fit_transform(x_embeddings)
#     x_test_reshaped = np.reshape(x_test, (len(x_test), 28 * 28))
    decomposed_gray = pca.fit_transform(x_embeddings_before_train)

    fig = plt.figure(figsize=(16, 8))
    for label in test_class_labels:
        decomposed_embeddings_class = decomposed_embeddings[y_test == label]
        decomposed_gray_class = decomposed_gray[y_test == label]

        plt.subplot(1,2,1)
        plt.scatter(decomposed_gray_class[::step,1], decomposed_gray_class[::step,0],label=str(label))
        plt.title('before training (embeddings)')
        plt.legend()

        plt.subplot(1,2,2)
        plt.scatter(decomposed_embeddings_class[::step, 1], decomposed_embeddings_class[::step, 0], label=str(label))
        plt.title('after @%d epochs' % epochs)
        plt.legend()

    #plt.show()
    fig.savefig('embeddings.png', bbox_inches='tight')
    return

def main(args):
    model_path = os.path.expanduser(args.model_path)
    weights_path = os.path.expanduser(args.weights_path)
    base_model_path = os.path.expanduser(args.base_model_path)
    base_weights_path = os.path.expanduser(args.base_weights_path)
    dir_path = os.path.expanduser(args.data_path)
    epochs = args.epochs
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    decay_rate = args.decay_rate
    decay_steps = args.decay_steps
    dim = (args.input_width, args.input_height)
    embedding_size = args.embedding_size 
    is_base_training = args.base_training_flag 
    is_base_testing = args.base_testing_flag 
    is_gpu = args.gpu

    if not is_gpu:
        tf.config.set_visible_devices([], 'GPU')

    # The data, split between train and test sets
    x_test = np.load("%s/test/x.npy" % (dir_path))
    y_test = np.load("%s/test/y.npy" % (dir_path))
    print(x_test.shape)
    print(y_test.shape)
    x_train = np.load("%s/train/x.npy" % (dir_path))
    y_train = np.load("%s/train/y.npy" % (dir_path))
    x_val = np.load("%s/validation/x.npy" % (dir_path))
    y_val = np.load("%s/validation/y.npy" % (dir_path))
    n_train = len(x_train)

    if not decay_steps:
        steps_per_epoch = n_train // batch_size
        decay_steps = steps_per_epoch * 100

    print("x_val.shape", x_val.shape)
    print("y_val.shape", y_val.shape)
    print("x_tarin.shape", x_train.shape)
    print("y_train.shape", y_train.shape)
    input_image_shape = (dim[0], dim[1], 3)
    if is_base_training or is_base_testing:
        model = train_base(
                is_base_training, model_path=base_model_path, weights_path=base_weights_path,
                input_image_shape=input_image_shape, embedding_size=embedding_size,
                batch_size=batch_size, epochs=epochs, lr=learning_rate, decay_rate=decay_rate,
                decay_steps=decay_steps, x_train=x_train, y_train=y_train,
                x_val=x_val, y_val=y_val, x_test=x_test, y_test=y_test)
        # Test the network
        # creating an empty network
        #before = base_model(input_image_shape, embedding_size=embedding_size)
        before = whole_model(input_image_shape, embedding_size=embedding_size)
    else:
        before = create_model_body(
                    base_model_path=base_model_path, base_weights_path=base_weights_path,
                    model_path=model_path, weights_path=weights_path, freeze=True)
        model = train_body(before,
               input_image_shape=input_image_shape, embedding_size=embedding_size,
               batch_size=batch_size, epochs=epochs, lr=learning_rate, decay_rate=decay_rate,
               decay_steps=decay_steps, x_train=x_train, y_train=y_train,
               x_val=x_val, y_val=y_val, x_test=x_test, y_test=y_test)
    test_trained_model(before, model, x_test, y_test, epochs)

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(
    description='Train Facenet model.')

    argparser.add_argument(
        '-m',
        '--model_path',
        help='path to HDF5 file containing a facenet model and weights',
        default='')

    argparser.add_argument(
        '-w',
        '--weights_path',
        help='path to previously trained facenet weights',
        default='')

    argparser.add_argument(
        '-bm',
        '--base_model_path',
        help='path to HDF5 file containing a facenet model and weights',
        default='')

    argparser.add_argument(
        '-bw',
        '--base_weights_path',
        help='path to previously trained facene_base weights',
        default='')

    argparser.add_argument(
        '-d',
        '--data_path',
        help='path to a directory containing numpy arrays (X and Y) of training dataset',
        default='dataset13/small')

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
        default=32)

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
        default=1)

    argparser.add_argument(
        '-ds',
        '--decay_steps',
        help='the decay steps',
        type=int,
        default=0)

    argparser.add_argument(
        '-iw',
        '--input_width',
        type=int,
        help='input image width',
        default=150)

    argparser.add_argument(
        '-ih',
        '--input_height',
        type=int,
        help='input image height',
        default=200)

    argparser.add_argument(
        '-em',
        '--embedding_size',
        type=int,
        help='embedding size',
        default=128)

    argparser.add_argument(
        '-btr',
        '--base_training_flag',
        help='base training flag',
        action='store_true',
        default=False)

    argparser.add_argument(
        '-btt',
        '--base_testing_flag',
        help='base testing flag',
        action='store_true',
        default=False)

    argparser.add_argument(
        '-g',
        '--gpu',
        help='use gpu',
        action='store_true',
        default=False)

    args = argparser.parse_args()
    main(args)
