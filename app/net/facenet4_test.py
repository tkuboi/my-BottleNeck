import sys
import pickle
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa

from tensorflow.keras.layers import *
from tensorflow.keras.models import load_model

from facenet4 import create_base_network2
from utils import import_images
from tripletloss_test import get_dists, get_neighbors, print_results

if __name__ == "__main__":
    if len(sys.argv) < 6:
        print("usage:<unique_dir> <test_dir> <model_file> <weight_file> <pickle_file>")
        exit()
    dir_path = sys.argv[1]
    test_dir_path = sys.argv[2]
    model_file = sys.argv[3]
    weight_file = sys.argv[4]
    pickle_file = sys.argv[5]

    #dim = (240, 320)
    dim = (150, 200)
    input_image_shape = (dim[0], dim[1], 3)
    embedding_size = 128

    x_train = np.load("%s/x.npy" % dir_path)
    y_train = np.load("%s/y.npy" % dir_path)
    print("Unique wines:", y_train.shape[0])

    x_test, y_test = import_images(test_dir_path, dim)
    x_test = np.array(x_test)
    y_test = np.array(y_test)

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255.
    x_test /= 255.

    #model = create_base_network2(input_image_shape, embedding_size)
    model = load_model(model_file)
    model.load_weights(weight_file)
    embeddings = model.predict(x_train)
    #print(embeddings)
    tests = model.predict(x_test)

    wines = pickle.load(open(pickle_file, 'rb'))
    results = get_neighbors(embeddings, y_train, tests, y_test, 10)
    print_results(results, y_test, wines)
    np.save('sample_wine_embeddings.npy', embeddings)
    np.save('sample_wine_ids.npy', y_train)
