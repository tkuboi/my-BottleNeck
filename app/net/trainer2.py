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

def read_images(dir_path, dimen, img_labels):
    files = os.listdir( dir_path )
    images = [] 
    labels = []
    for i, item in enumerate(files):
        file_path = os.path.join(dir_path , item)
        if os.path.isdir(file_path):
            continue
        #labels.append(item)
        #samples = []
        try:
            image = Image.open(file_path)
            resized_image = image.resize((dimen, dimen))
            arr = []
            for x in range(dimen):
                sub_array = []
                for y in range(dimen):
                    sub_array.append(resized_image.load()[x, y][:3])
                arr.append(sub_array)
            image_data = np.array(arr)
            image_data = np.array(np.reshape(image_data, (dimen, dimen, 3))) / 255
            labels.append(img_labels[item.lower()])
            images.append(image_data)
        except:
            traceback.print_exc(file=sys.stdout)
            print('WARNING : File {} could not be processed.'.format(file_path))
    return images, labels

def train_model(model, parameters, dim, images, labels):
    samples = []
    samples_1 = []
    samples_2 = []
    truth_values = [] #1:True, 0:False
    length = len(images)
    for i in range(length):
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
    samples = np.array(samples)
    X1 = X1.reshape((X1.shape[0], dim**2 * 3)).astype(np.float32)
    X2 = X2.reshape((X2.shape[0], dim**2 * 3)).astype(np.float32)
    samples = samples.reshape((samples.shape[0], dim**2 * 3)).astype(np.float32)
    Y = np.array(truth_values)
    model.fit([X1, X2], Y, hyperparameters=parameters)
    return model, samples 

def save_samples(samples, labels, dir_path):
    np.save('{}/x.npy'.format(dir_path), samples)
    with open('{}/labels.json'.format(dir_path), 'w') as json_file:
        json.dump(labels, json_file)

def main():
    if len(sys.argv) < 3:
        print("usage trainer.py <in_dir> <out_dir> [<dimen>]")
        exit()
    in_dir = sys.argv[1]
    out_dir = sys.argv[2]
    if len(sys.argv) > 3:
        dimen = sys.argv[3]
    else:
        dimen = DIMENSION

    img_labels = create_label_dict(in_dir+'/wineDataToImageFilename_2020_02_10.csv')
    images, labels = read_images(in_dir, dimen, img_labels)
    print(labels)
    parameters = {'batch_size':32, 'epochs':1, 'callbacks':None, 'val_data':None}
    model, samples = train_model(SiameseModel(), parameters, dimen, images, labels)
    model.save_model(out_dir + '/model.h5')
    save_samples(samples, labels, 'np_samples')

if __name__ == '__main__':
    main()
