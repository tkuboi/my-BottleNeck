import io
import os
import sys

from PIL import Image
from sklearn.decomposition import PCA

import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
#import tensorflow_datasets as tfds
import tensorflow_addons as tfa

def shuffle(X,Y):
    #shuffle examples and labels arrays together
    rng_state = np.random.get_state()
    np.random.shuffle(X)
    np.random.set_state(rng_state)
    np.random.shuffle(Y)

def read_images(dir_path, filename, dim):
    images = []
    labels = []
    count = 0
    file_path = os.path.join(dir_path, filename)
    with open(file_path) as f:
        for line in f.readlines():
            try:
                tokens = line.split(', ')
                image_path = os.path.join(dir_path, tokens[-1].replace('\n', ''))
                if not os.path.exists(image_path):
                    continue
                image = Image.open(image_path)
                image_data = to_np(image, dim)
                images.append(image_data)
                labels.append(int(tokens[0]))
                count += 1
            except:
                traceback.print_exc(file=sys.stdout)

    return np.array(images), np.array(labels)

def _normalize_img(img, label):
    img = tf.cast(img, tf.float32) / 255.
    return (img, label)

def create_model(dim):
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(filters=64, kernel_size=2, padding='same', activation='relu', input_shape=(dim,dim,3)),
        tf.keras.layers.MaxPooling2D(pool_size=2),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Conv2D(filters=32, kernel_size=2, padding='same', activation='relu'),
        tf.keras.layers.MaxPooling2D(pool_size=2),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(256, activation=None), # No activation on final dense layer
        tf.keras.layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1)) # L2 normalize embeddings

    ])

    return model

def plot_pca(testing_embeddings, x_test, y_test, x_embeddings_before_train,
        no_of_components, epochs):
    # Visualizing the effect of embeddings -> using PCA!
    x_embeddings = testing_embeddings.predict(x_test)
    step = 1
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

def transfer_weights(model, testing_embeddings):
    # Grabbing the weights from the trained network
    for layer_target, layer_source in zip(testing_embeddings.layers, model.layers):
        weights = layer_source.get_weights()
        layer_target.set_weights(weights)
        del weights

#train_dataset, test_dataset = tfds.load(name="mnist", split=['train', 'test'], as_supervised=True)
#print(train_dataset)

# Build your input pipelines
#train_dataset = train_dataset.shuffle(1024).batch(32)
#train_dataset = train_dataset.map(_normalize_img)

#test_dataset = test_dataset.batch(32)
#test_dataset = test_dataset.map(_normalize_img)
dim = 224
input_image_shape = (dim, dim, 3)
embedding_size = 256 
no_of_components = 2  # for visualization -> PCA.fit_transform()
epochs = 3

#(x_test, y_test) = read_images('testset', 'wineDataToImageFilename_2020_02_10.csv', dim)
x_test = np.load("np_test/x.npy") 
y_test = np.load("np_test/y.npy") 
print(x_test.shape)
print(y_test.shape)
#(x_train, y_train) = read_images('dataset4', 'wineDataToImageFilename_2020_02_10.csv', dim)
x_train = np.load("np_train/x.npy") 
y_train = np.load("np_train/y.npy")
x_train = x_train[:400]
y_train = y_train[:400]
shuffle(x_train, y_train)
x_val = x_train[-10:,:,:]
y_val = y_train[-10:]
#x_train = x_train[:-10,:,:]
#y_train = y_train[:-10]
n_train = len(x_train)
#steps_per_epoch = n_train // batch_size
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

train_dataset = (x_train, y_train)
test_dataset = (x_val, y_val)

model = create_model(dim)
# Compile the model
model.compile(
    optimizer=tf.keras.optimizers.Adam(0.0001),
    loss=tfa.losses.TripletSemiHardLoss())

# Train the network
history = model.fit(
    x_train,
    y_train,
    batch_size=100,
    epochs=epochs,
    validation_data=(x_val, y_val))

fig = plt.figure(figsize=(8,8))
plt.plot(history.history['loss'], label='training loss')
plt.plot(history.history['val_loss'], label='validation loss')
plt.legend()
plt.title('Train/validation loss')
plt.show()
fig.savefig('train_val_loss.png', bbox_inches='tight')

# Evaluate the network
results = model.predict(x_test)

# Save test embeddings for visualization in projector
np.savetxt("vecs.tsv", results, delimiter='\t')

# Test the network
# creating an empty network
testing_embeddings = create_model(dim)
x_embeddings_before_train = testing_embeddings.predict(x_test)
transfer_weights(model, testing_embeddings)
plot_pca(testing_embeddings, x_test, y_test, x_embeddings_before_train,
         no_of_components, epochs)

#out_m = io.open('meta.tsv', 'w', encoding='utf-8')
#for img, labels in tfds.as_numpy(test_dataset):
#    [out_m.write(str(x) + "\n") for x in labels]
#out_m.close()


#try:
#  from google.colab import files
#  files.download('vecs.tsv')
#  files.download('meta.tsv')
#except:
#  pass


