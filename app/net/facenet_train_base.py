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

from sklearn.decomposition import PCA

from PIL import Image

from inception import base_model

if __name__ == "__main__":
    # in case this scriot is called from another file, let's make sure it doesn't start training the network...
    if len(sys.argv) < 3: 
        print("Usage: <dir_path> <epochs> <optional:weight_path> <optional:train_flag>")
        exit()
    dir_path = sys.argv[1]
    epochs = int(sys.argv[2]) 
    weight_file = sys.argv[3] if len(sys.argv) > 3 else None
    train_flag_str = sys.argv[4] if len(sys.argv) > 4 else '1'
    dim = (150, 200) 
    batch_size = 32 
    train_flag = True if train_flag_str == '1' else False  # either True or False

    #embedding_size = 64
    embedding_size = 128 

    no_of_components = 2  # for visualization -> PCA.fit_transform()

    step = 1

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
    steps_per_epoch = n_train // batch_size
    print("x_val.shape", x_val.shape)
    print("y_val.shape", y_val.shape)
    print("x_tarin.shape", x_train.shape)
    print("y_train.shape", y_train.shape)
    input_image_shape = (dim[0], dim[1], 3)

    # Network training...
    if train_flag == True:
        model = base_model(input_image_shape, embedding_size)
        plot_model(model, to_file='base_network.png', show_shapes=True, show_layer_names=True)
        input_images = Input(shape=input_image_shape, name='input_image') # input layer for images
        input_labels = Input(shape=(1,), name='input_label')    # input layer for labels
        embeddings = model([input_images])               # output of network -> embeddings
        #labels_plus_embeddings = concatenate([input_labels, embeddings])  # concatenating the labels + embeddings

        model.summary()
        model.save('saved_model/facenet_base_model')
        plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=True)

        lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(
                          0.0001,
                          decay_steps=steps_per_epoch*100,
                          decay_rate=1,
                          staircase=False)
        # train session
        #opt = Adam(lr=0.0001)  # choose optimiser. RMS is good too!
        opt = Adam(lr_schedule)  # choose optimiser. RMS is good too!
        #opt = Adagrad(lr=0.00001)  # choose optimiser. RMS is good too!
        #opt = Adagrad(lr_schedule)  # choose optimiser. RMS is good too!
        #opt = RMSprop(lr_schedule)  # choose optimiser. RMS is good too!

        #model.compile(loss=triplet_loss_adapted_from_tf, optimizer=opt)
        model.compile(loss=tfa.losses.TripletSemiHardLoss(), optimizer=opt)

        if weight_file:
            model.load_weights(weight_file)

        checkpoint_path = "facenet_base_ep{epoch:02d}_BS%d/cp.ckpt" % batch_size
        checkpoint_dir = os.path.dirname(checkpoint_path)
        #checkpoint = ModelCheckpoint(filepath, verbose=1, save_best_only=False, period=5)
        checkpoint = ModelCheckpoint(checkpoint_path, verbose=1, save_best_only=True, save_weights_only=True, period=1)
        callbacks_list = [checkpoint]

        # Uses 'dummy' embeddings + dummy gt labels. Will be removed as soon as loaded, to free memory
        dummy_gt_train = np.zeros((len(x_train), embedding_size + 1))
        dummy_gt_val = np.zeros((len(x_val), embedding_size + 1))

        print(x_train.shape)
        print(x_val.shape)
        H = model.fit(
            x=x_train,
            y=y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(x_val, y_val),
            callbacks=callbacks_list)

        model.save_weights(checkpoint_path.format(epoch=epochs))

        fig = plt.figure(figsize=(8,8))
        plt.plot(H.history['loss'], label='training loss')
        plt.plot(H.history['val_loss'], label='validation loss')
        plt.legend()
        plt.title('Train/validation loss')
        plt.show()
        fig.savefig('train_val_loss.png', bbox_inches='tight')
    else:

        #####
        model = load_model('saved_model/facenet_base_model')
        model.load_weights(weight_file)

    # Test the network
    # creating an empty network
    testing_embeddings = base_model(input_image_shape,
                                             embedding_size=embedding_size)
    x_embeddings_before_train = testing_embeddings.predict(x_test)
    # Grabbing the weights from the trained network
    for layer_target, layer_source in zip(testing_embeddings.layers, model.layers):
        weights = layer_source.get_weights()
        layer_target.set_weights(weights)
        del weights

    # Visualizing the effect of embeddings -> using PCA!
    x_embeddings = testing_embeddings.predict(x_test)
    dict_embeddings = {}
    dict_gray = {}
    test_class_labels = np.unique(y_test)

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

