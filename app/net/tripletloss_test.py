import io
import os
import sys
import pickle

from PIL import Image
from sklearn.decomposition import PCA

from tensorflow.keras.callbacks import ModelCheckpoint

import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
#import tensorflow_datasets as tfds
import tensorflow_addons as tfa



def create_model(dim, embedding_size):
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(filters=64, kernel_size=2, padding='same', activation='relu', input_shape=(dim[0],dim[1],3)),
        tf.keras.layers.MaxPooling2D(pool_size=2),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Conv2D(filters=32, kernel_size=2, padding='same', activation='relu'),
        tf.keras.layers.MaxPooling2D(pool_size=2),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(embedding_size, activation=None), # No activation on final dense layer
        tf.keras.layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1)) # L2 normalize embeddings

    ])

    return model

def get_dists(embeddings, test):
    print(len(embeddings))
    dists = [None] * len(embeddings)
    for i, embedding in enumerate(embeddings):
        #print(embedding)
        #print(test)
        dists[i] = np.linalg.norm(embedding - test) 
    return dists
 
def get_neighbors(embeddings, y_train, tests, y_test, k):
    #print(set(y_train))
    results = []
    for i, test in enumerate(tests):
        dists = get_dists(embeddings, test)
        zipped = zip(dists, y_train)
        zipped = list(zipped)
        #print(zipped)
        neighbors = sorted(zipped, key=lambda x : x[0])
        results.append(neighbors[:k])
    return results

def print_results(results, y_test, wines):
    #print(results)
    for i, neighbors in enumerate(results):
        _id = int(y_test[i])
        print("-" * 20)
        print(wines[_id] if _id in wines else _id, " is one of :")
        for neighbor in neighbors:
            n_id = int(neighbor[1])
            if n_id in wines:
                print(n_id, wines[n_id], ", distance: ", neighbor[0])
            else:
                print(n_id, ", confidence: ", neighbor[0])
                
        print("-" * 20)
 
def main():
    if len(sys.argv) < 4:
        print("usage:python tripletloss_test.py unique_dir test_dir weight_file pickle_file")
        exit()
    dir_path = sys.argv[1]
    test_dir_path = sys.argv[2]
    weight_file = sys.argv[3]
    pickle_file = sys.argv[4]

    dim = (240, 320)
    input_image_shape = (dim[0], dim[1], 3)
    embedding_size = 128

    x_train = np.load("%s/x.npy" % dir_path)
    y_train = np.load("%s/y.npy" % dir_path)

    x_test = np.load("%s/test/x.npy" % test_dir_path)
    y_test = np.load("%s/test/y.npy" % test_dir_path)

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255.
    x_test /= 255.

    model = create_model(dim, embedding_size)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(0.0001),
        loss=tfa.losses.TripletSemiHardLoss())
  
    model.load_weights(weight_file)

    embeddings = model.predict(x_train)
    #print(embeddings)
    tests = model.predict(x_test)

    wines = pickle.load(open(pickle_file, 'rb'))
    results = get_neighbors(embeddings, y_train, tests, y_test, 10)
    print_results(results, y_test, wines)

if __name__ == '__main__':
    main()
