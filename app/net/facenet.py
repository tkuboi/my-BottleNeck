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

def pairwise_distance(feature, squared=False):
    """Computes the pairwise distance matrix with numerical stability.

    output[i, j] = || feature[i, :] - feature[j, :] ||_2

    Args:
      feature: 2-D Tensor of size [number of data, feature dimension].
      squared: Boolean, whether or not to square the pairwise distances.

    Returns:
      pairwise_distances: 2-D Tensor of size [number of data, number of data].
    """
    pairwise_distances_squared = math_ops.add(
        math_ops.reduce_sum(math_ops.square(feature), axis=[1], keepdims=True),
        math_ops.reduce_sum(
            math_ops.square(tf.transpose(feature)),
            axis=[0],
            keepdims=True))\
        - 2.0 * tf.linalg.matmul(feature, tf.transpose(feature))

    # Deal with numerical inaccuracies. Set small negatives to zero.
    pairwise_distances_squared = math_ops.maximum(pairwise_distances_squared, 0.0)
    # Get the mask where the zero distances are at.
    error_mask = math_ops.less_equal(pairwise_distances_squared, 0.0)

    # Optionally take the sqrt.
    if squared:
        pairwise_distances = pairwise_distances_squared
    else:
        pairwise_distances = math_ops.sqrt(
            pairwise_distances_squared + tf.compat.v1.to_float(error_mask) * 1e-16)

    # Undo conditionally adding 1e-16.
    pairwise_distances = math_ops.multiply(
        pairwise_distances, tf.compat.v1.to_float(math_ops.logical_not(error_mask)))

    num_data = tf.shape(feature)[0]
    # Explicitly set diagonals to zero.
    mask_offdiagonals = tf.ones_like(pairwise_distances) - tf.linalg.diag(
        tf.ones([num_data]))
    pairwise_distances = math_ops.multiply(pairwise_distances, mask_offdiagonals)
    return pairwise_distances

def masked_maximum(data, mask, dim=1):
    """Computes the axis wise maximum over chosen elements.

    Args:
      data: 2-D float `Tensor` of size [n, m].
      mask: 2-D Boolean `Tensor` of size [n, m].
      dim: The dimension over which to compute the maximum.

    Returns:
      masked_maximums: N-D `Tensor`.
        The maximized dimension is of size 1 after the operation.
    """
    axis_minimums = math_ops.reduce_min(data, dim, keepdims=True)
    masked_maximums = math_ops.reduce_max(
        math_ops.multiply(data - axis_minimums, mask), dim,
        keepdims=True) + axis_minimums
    return masked_maximums

def masked_minimum(data, mask, dim=1):
    """Computes the axis wise minimum over chosen elements.

    Args:
      data: 2-D float `Tensor` of size [n, m].
      mask: 2-D Boolean `Tensor` of size [n, m].
      dim: The dimension over which to compute the minimum.

    Returns:
      masked_minimums: N-D `Tensor`.
        The minimized dimension is of size 1 after the operation.
    """
    axis_maximums = math_ops.reduce_max(data, dim, keepdims=True)
    masked_minimums = math_ops.reduce_min(
        math_ops.multiply(data - axis_maximums, mask), dim,
        keepdims=True) + axis_maximums
    return masked_minimums

def triplet_loss_adapted_from_tf(y_true, y_pred):
    del y_true
    margin = 1.
    labels = y_pred[:, :1]

 
    labels = tf.cast(labels, dtype='int32')

    embeddings = y_pred[:, 1:]
    #tf.print("feature", embeddings, output_stream=sys.stdout)

    ### Code from Tensorflow function [tf.contrib.losses.metric_learning.triplet_semihard_loss] starts here:
    
    # Reshape [batch_size] label tensor to a [batch_size, 1] label tensor.
    # lshape=tf.shape(labels)
    # assert lshape.shape == 1
    # labels = tf.reshape(labels, [lshape[0], 1])

    # Build pairwise squared distance matrix.
    #pdist_matrix = pairwise_distance(embeddings, squared=False)
    pdist_matrix = tfa.losses.metric_learning.pairwise_distance(embeddings)
    # Build pairwise binary adjacency matrix.
    adjacency = math_ops.equal(labels, tf.transpose(labels))
    # Invert so we can select negatives only.
    adjacency_not = math_ops.logical_not(adjacency)

    # global batch_size  
    batch_size = tf.size(labels) # was 'tf.size(labels)'

    # Compute the mask.
    pdist_matrix_tile = tf.tile(pdist_matrix, [batch_size, 1])
    mask = math_ops.logical_and(
        tf.tile(adjacency_not, [batch_size, 1]),
        math_ops.greater(
            pdist_matrix_tile, tf.reshape(
                tf.transpose(pdist_matrix), [-1, 1])))
    mask_final = tf.reshape(
        math_ops.greater(
            math_ops.reduce_sum(
                tf.cast(mask, dtype=dtypes.float32), 1, keepdims=True),
            0.0), [batch_size, batch_size])
    mask_final = tf.transpose(mask_final)

    adjacency_not = tf.cast(adjacency_not, dtype=dtypes.float32)
    mask = tf.cast(mask, dtype=dtypes.float32)

    # negatives_outside: smallest D_an where D_an > D_ap.
    negatives_outside = tf.reshape(
        masked_minimum(pdist_matrix_tile, mask), [batch_size, batch_size])
    negatives_outside = tf.transpose(negatives_outside)

    # negatives_inside: largest D_an.
    negatives_inside = tf.tile(
        masked_maximum(pdist_matrix, adjacency_not), [1, batch_size])
    semi_hard_negatives = tf.where(
        mask_final, negatives_outside, negatives_inside)

    loss_mat = math_ops.add(margin, pdist_matrix - semi_hard_negatives)

    mask_positives = tf.cast(
        adjacency, dtype=dtypes.float32) - tf.linalg.diag(
        tf.ones([batch_size]))

    # In lifted-struct, the authors multiply 0.5 for upper triangular
    #   in semihard, they take all positive pairs except the diagonal.
    num_positives = math_ops.maximum(1.0, math_ops.reduce_sum(mask_positives))

    semi_hard_triplet_loss_distance = math_ops.truediv(
        math_ops.reduce_sum(
            math_ops.maximum(
                math_ops.multiply(loss_mat, mask_positives), 0.0)),
        num_positives,
        name='triplet_semihard_loss')

    ### Code from Tensorflow function semi-hard triplet loss ENDS here.
    return semi_hard_triplet_loss_distance

def create_base_network(image_input_shape, embedding_size):
    """
    Base network to be shared (eq. to feature extraction).
    """
    input_image = Input(shape=image_input_shape)

    x = Flatten()(input_image)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.1)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.1)(x)
    x = Dense(embedding_size)(x)

    base_network = Model(inputs=input_image, outputs=x)
    plot_model(base_network, to_file='base_network.png', show_shapes=True, show_layer_names=True)
    return base_network

def create_base_network2(image_input_shape, embedding_size):
    """
    Base network to be shared (eq. to feature extraction).
    """
    input_image = Input(shape=image_input_shape)

    #x = Flatten()(input_image)
    #x = Conv2D(64, kernel_size=(7, 7), strides=2, padding='same', activation='relu', kernel_regularizer=regularizers.l2(0.001))(input_image)
    x = Conv2D(64, kernel_size=(7, 7), strides=2, padding='same', activation='relu' )(input_image)
    x = MaxPooling2D(pool_size=(3, 3), strides=2, padding='same')(x)
    #x = LayerNormalization(epsilon=1e-6)(x)
    x = l2_normalize(x, 0)
    #x = MaxPooling2D(pool_size=(3, 3), strides=2, padding='same')(x)
    #inception 2
    x = Conv2D(64, kernel_size=(1, 1), strides=1, padding='same', activation='relu')(x)
    x = Conv2D(192, kernel_size=(3, 3), strides=1, padding='same', activation='relu')(x)
    #x = concatenate([x, x1], axis=3)
    #x = LayerNormalization(epsilon=1e-6)(x)
    x = l2_normalize(x, 0)
    x = MaxPooling2D(pool_size=(3, 3), strides=2, padding='same')(x)

    #3a
    x1 = Conv2D(64, kernel_size=(1, 1), strides=1, padding='same', activation='relu')(x)
    x2 = Conv2D(96, kernel_size=(1, 1), strides=1, padding='same', activation='relu')(x)
    x2 = Conv2D(128, kernel_size=(3, 3), strides=1, padding='same', activation='relu')(x2)
    x3 = Conv2D(16, kernel_size=(1, 1), strides=1, padding='same', activation='relu')(x)
    x3 = Conv2D(32, kernel_size=(5, 5), strides=1, padding='same', activation='relu')(x3)
    x4 = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(x)
    x4 = Conv2D(32, kernel_size=(1, 1), strides=1, padding='same', activation='relu')(x4)
    x = concatenate([x1, x2, x3, x4], axis=3)
    x = Dropout(0.2)(x)

    #3b
    x1 = Conv2D(64, kernel_size=(1, 1), strides=1, padding='same', activation='relu')(x)
    x2 = Conv2D(96, kernel_size=(1, 1), strides=1, padding='same', activation='relu')(x)
    x2 = Conv2D(128, kernel_size=(3, 3), strides=1, padding='same', activation='relu')(x2)
    x3 = Conv2D(32, kernel_size=(1, 1), strides=1, padding='same', activation='relu')(x)
    x3 = Conv2D(64, kernel_size=(5, 5), strides=1, padding='same', activation='relu')(x3)
    x4 = l2_normalize(x, 0) 
    x4 = Conv2D(64, kernel_size=(1, 1), strides=1, padding='same', activation='relu')(x4)
    x = concatenate([x1, x2, x3, x4], axis=3)
    x = Dropout(0.2)(x)

    #3c
    x2 = Conv2D(128, kernel_size=(1, 1), strides=1, padding='same', activation='relu')(x)
    x2 = Conv2D(256, kernel_size=(3, 3), strides=2, padding='same', activation='relu')(x2)
    x3 = Conv2D(32, kernel_size=(1, 1), strides=1, padding='same', activation='relu')(x)
    x3 = Conv2D(64, kernel_size=(5, 5), strides=2, padding='same', activation='relu')(x3)
    x4 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)
    x = concatenate([x2, x3, x4], axis=3)
    x = Dropout(0.2)(x)

    #4a
    x1 = Conv2D(256, kernel_size=(1, 1), strides=1, padding='same', activation='relu')(x)
    x2 = Conv2D(96, kernel_size=(1, 1), strides=1, padding='same', activation='relu')(x)
    x2 = Conv2D(192, kernel_size=(3, 3), strides=1, padding='same', activation='relu')(x2)
    x3 = Conv2D(32, kernel_size=(1, 1), strides=1, padding='same', activation='relu')(x)
    x3 = Conv2D(64, kernel_size=(5, 5), strides=1, padding='same', activation='relu')(x3)
    x4 = l2_normalize(x, 0) 
    x4 = Conv2D(128, kernel_size=(1, 1), strides=1, padding='same', activation='relu')(x4)
    x = concatenate([x1, x2, x3, x4], axis=3)
    x = Dropout(0.2)(x)

    #4b
    x1 = Conv2D(224, kernel_size=(1, 1), strides=1, padding='same', activation='relu')(x)
    x2 = Conv2D(112, kernel_size=(1, 1), strides=1, padding='same', activation='relu')(x)
    x2 = Conv2D(224, kernel_size=(3, 3), strides=1, padding='same', activation='relu')(x2)
    x3 = Conv2D(32, kernel_size=(1, 1), strides=1, padding='same', activation='relu')(x)
    x3 = Conv2D(64, kernel_size=(5, 5), strides=1, padding='same', activation='relu')(x3)
    x4 = l2_normalize(x, 0) 
    x4 = Conv2D(128, kernel_size=(1, 1), strides=1, padding='same', activation='relu')(x4)
    x = concatenate([x1, x2, x3, x4], axis=3)
    x = Dropout(0.2)(x)

    #4c
    x1 = Conv2D(192, kernel_size=(1, 1), strides=1, padding='same', activation='relu')(x)
    x2 = Conv2D(128, kernel_size=(1, 1), strides=1, padding='same', activation='relu')(x)
    x2 = Conv2D(256, kernel_size=(3, 3), strides=1, padding='same', activation='relu')(x2)
    x3 = Conv2D(32, kernel_size=(1, 1), strides=1, padding='same', activation='relu')(x)
    x3 = Conv2D(64, kernel_size=(5, 5), strides=1, padding='same', activation='relu')(x3)
    x4 = l2_normalize(x, 0) 
    x4 = Conv2D(128, kernel_size=(1, 1), strides=1, padding='same', activation='relu')(x4)
    x = concatenate([x1, x2, x3, x4], axis=3)
    x = Dropout(0.2)(x)

    #4d
    x1 = Conv2D(160, kernel_size=(1, 1), strides=1, padding='same', activation='relu')(x)
    x2 = Conv2D(144, kernel_size=(1, 1), strides=1, padding='same', activation='relu')(x)
    x2 = Conv2D(288, kernel_size=(3, 3), strides=1, padding='same', activation='relu')(x2)
    x3 = Conv2D(32, kernel_size=(1, 1), strides=1, padding='same', activation='relu')(x)
    x3 = Conv2D(64, kernel_size=(5, 5), strides=1, padding='same', activation='relu')(x3)
    x4 = l2_normalize(x, 0) 
    x4 = Conv2D(128, kernel_size=(1, 1), strides=1, padding='same', activation='relu')(x4)
    x = concatenate([x1, x2, x3, x4], axis=3)
    x = Dropout(0.2)(x)

    #4e
    x2 = Conv2D(160, kernel_size=(1, 1), strides=1, padding='same', activation='relu')(x)
    x2 = Conv2D(256, kernel_size=(3, 3), strides=2, padding='same', activation='relu')(x2)
    x3 = Conv2D(64, kernel_size=(1, 1), strides=1, padding='same', activation='relu')(x)
    x3 = Conv2D(128, kernel_size=(5, 5), strides=2, padding='same', activation='relu')(x3)
    x4 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)
    x = concatenate([x2, x3, x4], axis=3)
    x = Dropout(0.2)(x)

    #5a
    x1 = Conv2D(384, kernel_size=(1, 1), strides=1, padding='same', activation='relu')(x)
    x2 = Conv2D(192, kernel_size=(1, 1), strides=1, padding='same', activation='relu')(x)
    x2 = Conv2D(384, kernel_size=(3, 3), strides=1, padding='same', activation='relu')(x2)
    x3 = Conv2D(48, kernel_size=(1, 1), strides=1, padding='same', activation='relu')(x)
    x3 = Conv2D(128, kernel_size=(5, 5), strides=1, padding='same', activation='relu')(x3)
    x4 = l2_normalize(x, 0) 
    x4 = Conv2D(128, kernel_size=(1, 1), strides=1, padding='same', activation='relu')(x4)
    x = concatenate([x1, x2, x3, x4], axis=3)
    x = Dropout(0.2)(x)

    #5b
    x1 = Conv2D(384, kernel_size=(1, 1), strides=1, padding='same', activation='relu')(x)
    x2 = Conv2D(192, kernel_size=(1, 1), strides=1, padding='same', activation='relu')(x)
    x2 = Conv2D(384, kernel_size=(3, 3), strides=1, padding='same', activation='relu')(x2)
    x3 = Conv2D(48, kernel_size=(1, 1), strides=1, padding='same', activation='relu')(x)
    x3 = Conv2D(128, kernel_size=(5, 5), strides=1, padding='same', activation='relu')(x3)
    x4 = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(x)
    x4 = Conv2D(128, kernel_size=(1, 1), strides=1, padding='same', activation='relu')(x4)
    x = concatenate([x1, x2, x3, x4], axis=3)
    x = Dropout(0.2)(x)


    x = AveragePooling2D(pool_size=(7, 7), strides=1)(x)
    #x = AveragePooling2D(pool_size=(2, 2), strides=1)(x)
    x = Flatten()(x)
    #x = Dense(128, activation='relu')(x)
    #x = Dropout(0.2)(x)
    x = Dense(embedding_size, activation='relu')(x)
    #x = LayerNormalization(epsilon=1e-6)(x)
    x = l2_normalize(x, 0)

    base_network = Model(inputs=input_image, outputs=x)
    plot_model(base_network, to_file='base_network.png', show_shapes=True, show_layer_names=True)
    return base_network

def read_images(dir_path, filename, dim):
    images = []
    labels = []
    count = 0
    file_path = os.path.join(dir_path, filename)
    with open(file_path) as f:
        for line in f.readlines():
            try:
                tokens = line.split(', ')
                #if extract_imgnum(tokens[-1]) == 1:
                #    continue
                image_path = os.path.join(dir_path, tokens[-1].replace('\n', ''))
                if not os.path.exists(image_path):
                    continue
                image = Image.open(image_path)
                image_data = to_np(image, dim)
                images.append(image_data)
                labels.append(int(tokens[0]))
                print(int(tokens[0]), image_path)
                count += 1
            except:
                traceback.print_exc(file=sys.stdout)

    return np.array(images), np.array(labels)

def to_np(image, dim):
    arr = []
    w, h = image.size
    for x in range(w):
        sub_array = []
        for y in range(h):
            sub_array.append(image.load()[x, y][:3])
        arr.append(sub_array)
    image_data = np.array(arr)
    image_data = np.array(np.reshape(image_data, (dim, dim, 3)))
    return image_data

def np_save(x, y, dir_path):
    np.save('{}/x.npy'.format(dir_path), x)
    np.save('{}/y.npy'.format(dir_path), y)

def extract_imgnum(filename):
    tokens = filename.split('.')
    parts = tokens[0].split('_')
    return parts[-1]

def shuffle(X,Y):
    #shuffle examples and labels arrays together
    rng_state = np.random.get_state()
    np.random.shuffle(X)
    np.random.set_state(rng_state)
    np.random.shuffle(Y)

if __name__ == "__main__":
    # in case this scriot is called from another file, let's make sure it doesn't start training the network...

    dim = 224 
    batch_size = 312 
    epochs = 10 
    train_flag = True  # either     True or False

    #embedding_size = 64
    embedding_size = 128 

    no_of_components = 2  # for visualization -> PCA.fit_transform()

    step = 1

    # The data, split between train and test sets
    #(x_train, y_train), (x_test, y_test) = mnist.load_data()
    #(x_test, y_test) = read_images('testset2', 'wineDataToImageFilename_2020_02_10.csv', dim)
    #np_save(x_test, y_test, "np_test")
    x_test = np.load("np_test/x.npy")
    y_test = np.load("np_test/y.npy")
    print(x_test.shape)
    print(y_test.shape)
    #(x_train, y_train) = read_images('dataset5', 'wineDataToImageFilename_2020_02_10.csv', dim)
    #np_save(x_train, y_train, "np_train")
    x_train = np.load("np_train/x.npy")[:400]
    y_train = np.load("np_train/y.npy")[:400]
    shuffle(x_train, y_train)
    x_val = x_train[-100:,:,:]
    y_val = y_train[-100:]
    #x_train = x_train[:-100,:,:]
    #y_train = y_train[:-100]
    n_train = len(x_train)
    steps_per_epoch = n_train // batch_size
    print("x_val.shape", x_val.shape)
    print("y_val.shape", y_val.shape)
    print("x_tarin.shape", x_train.shape)
    print("y_train.shape", y_train.shape)
    x_train = x_train.astype('float32')
    x_val = x_val.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255.
    x_val /= 255.
    x_test /= 255.
    #input_image_shape = (28, 28, 1)
    input_image_shape = (dim, dim, 3)

    # Network training...
    if train_flag == True:
        base_network = create_base_network2(input_image_shape, embedding_size)

        input_images = Input(shape=input_image_shape, name='input_image') # input layer for images
        input_labels = Input(shape=(1,), name='input_label')    # input layer for labels
        embeddings = base_network([input_images])               # output of network -> embeddings
        labels_plus_embeddings = concatenate([input_labels, embeddings])  # concatenating the labels + embeddings

        # Defining a model with inputs (images, labels) and outputs (labels_plus_embeddings)
        model = Model(inputs=[input_images, input_labels],
                      outputs=labels_plus_embeddings)

        model.summary()
        plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=True)

        lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(
                          0.0001,
                          decay_steps=steps_per_epoch*1000,
                          decay_rate=1,
                          staircase=False)
        # train session
        #opt = Adam(lr=0.0001)  # choose optimiser. RMS is good too!
        #opt = Adagrad(lr=0.00001)  # choose optimiser. RMS is good too!
        opt = RMSprop(lr_schedule)  # choose optimiser. RMS is good too!

        model.compile(loss=triplet_loss_adapted_from_tf, optimizer=opt)
        #model.compile(loss=tfa.losses.TripletSemiHardLoss(), optimizer=opt)

        filepath = "semiH_trip_MNIST_v13_ep{epoch:02d}_BS%d.hdf5" % batch_size
        checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=False, period=25)
        callbacks_list = [checkpoint]

        # Uses 'dummy' embeddings + dummy gt labels. Will be removed as soon as loaded, to free memory
        dummy_gt_train = np.zeros((len(x_train), embedding_size + 1))
        dummy_gt_val = np.zeros((len(x_val), embedding_size + 1))

        #x_train = np.reshape(x_train, (len(x_train), x_train.shape[1], x_train.shape[1], 3))
        #x_val = np.reshape(x_val, (len(x_val), x_train.shape[1], x_train.shape[1], 3))

        print(x_train.shape)
        print(x_val.shape)
        H = model.fit(
            x=[x_train,y_train],
            y=dummy_gt_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=([x_val, y_val], dummy_gt_val),
            callbacks=callbacks_list)

        plt.figure(figsize=(8,8))
        plt.plot(H.history['loss'], label='training loss')
        plt.plot(H.history['val_loss'], label='validation loss')
        plt.legend()
        plt.title('Train/validation loss')
        plt.show()
    else:

        #####
        model = load_model('semiH_trip_MNIST_v13_ep25_BS256.hdf5',
                                        custom_objects={'triplet_loss_adapted_from_tf':triplet_loss_adapted_from_tf})

    # Test the network
    # creating an empty network
    testing_embeddings = create_base_network2(input_image_shape,
                                             embedding_size=embedding_size)
    x_embeddings_before_train = testing_embeddings.predict(x_test)
    # Grabbing the weights from the trained network
    for layer_target, layer_source in zip(testing_embeddings.layers, model.layers[2].layers):
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

    plt.show()

