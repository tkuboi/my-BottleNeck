"""Data Preprocessing program

Author:
    Toshi Kuboi
"""

import numpy as np
import os
import sys
import traceback
import json
from tensorflow import keras
from PIL import Image
from ann import SiameseModel
from crop_image import crop_image

DIMENSION = 128

def create_label_dict(file_path):
    filename_label = dict()
    with open(file_path) as csv:
        i = 0
        line = csv.readline()
        while line:
            if i != 0:
                items = line.split(',')
                filename = items[-1].split('/')[-1]
                filename = filename.replace('\n', '')
                filename_label[filename] = "%s %s" % (items[2], items[3])
                print(filename)
            line = csv.readline()
            i += 1
    return filename_label

def read_images(dir_path, dim, img_labels):
    files = os.listdir( dir_path )
    images = [] 
    labels = []
    count = 0
    for i, item in enumerate(files):
        file_path = os.path.join(dir_path , item)
        if os.path.isdir(file_path):
            continue
        #labels.append(item)
        #samples = []
        try:
            image = Image.open(file_path)
            cropped = crop_image(image)
            resized_images = [image.resize((dim, dim)) for image in cropped]
            for resized_image in resized_images:
                #resized_image.save("cropped/new_" + item)
                image_data = to_np(resized_image, dim)
                labels.append(img_labels[item.lower()])
                images.append(image_data)
            count += 1
        except:
            traceback.print_exc(file=sys.stdout)
            print('WARNING : File {} could not be processed.'.format(file_path))
    return images, labels

def to_np(image, dim):
    arr = []
    for x in range(dim):
        sub_array = []
        for y in range(dim):
            sub_array.append(image.load()[x, y][:3])
        arr.append(sub_array)
    image_data = np.array(arr)
    image_data = np.array(np.reshape(image_data, (dim, dim, 3))) / 255
    return image_data 

def train_model(model, parameters, dim, images, labels, offset=0, step=10):
    samples = []
    samples_1 = []
    samples_2 = []
    truth_values = [] #1:True, 0:False
    length = len(images)
    to = min(length, offset + step)
    for i in range(offset, to):
        samples.append(images[i])
        for j in range(length):
            samples_1.append(images[i])
            samples_2.append(images[j])
            if labels[i] == labels[j]:
                truth_values.append(1)
            else:
                truth_values.append(0)
    X1 = np.array(samples_1) 
    X2 = np.array(samples_2)
    #samples = np.array(samples)
    X1 = X1.reshape((X1.shape[0], dim**2 * 3)).astype(np.float32)
    X2 = X2.reshape((X2.shape[0], dim**2 * 3)).astype(np.float32)
    #samples = samples.reshape((samples.shape[0], dim**2 * 3)).astype(np.float32)
    Y = np.array(truth_values)
    model.fit([X1, X2], Y, hyperparameters=parameters)
    return model, samples 

def save_samples(samples, labels, dim, dir_path, batch_num=None):
    samples = np.array(samples)
    samples = samples.reshape((samples.shape[0], dim**2 * 3)).astype(np.float32)
    if batch_num is None:
        np.save('{}/x.npy'.format(dir_path), samples)
    else:
        np.save('{}/x_{}.npy'.format(dir_path, batch_num), samples)
    json_filename = '{}/labels.json'.format(dir_path) if batch_num is None\
        else '{}/labels_{}.json'.format(dir_path, batch_num)
    with open(json_filename, 'w') as json_file:
        json.dump(labels, json_file)

def main():
    if len(sys.argv) < 3:
        print("usage trainer.py <in_dir> <out_dir> [<dim>]")
        exit()
    in_dir = sys.argv[1]
    out_dir = sys.argv[2]
    if len(sys.argv) > 3:
        dim = sys.argv[3]
    else:
        dim = DIMENSION

    img_labels = create_label_dict(in_dir+'/wineDataToImageFilename_2020_02_10.csv')
    images, labels = read_images(in_dir, dim, img_labels)
    for i, label in enumerate(labels):
        if "J. Lohr" in label:
            print(i, label)
    images, labels = images[6600:7200], labels[6600:7200]
    save_samples(images, labels, dim, 'np_samples')
    print(labels)
    parameters = {'batch_size':32, 'epochs':1, 'callbacks':None, 'val_data':None}
    length = len(images)
    print(length)
    batch = 10
    _model = SiameseModel()
    for i in range(0, length, batch):
        print("batch ", i)
        model, samples = train_model(_model, parameters, dim, images, labels, i, batch)
    model.save_model(out_dir + '/model.h5')

if __name__ == '__main__':
    main()
