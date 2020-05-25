import os
import sys
import traceback
import numpy as np
from PIL import Image

DIMENSION = (240, 320)

def import_images(filepath, dim):
    dir_path = "/".join(filepath.split("/")[:-1])
    dataset = {"train":{"x":[], "y":[]}, "test":{"x":[], "y":[]}, "validation":{"x":[], "y":[]}}
    data = []
    labels = []
    wine = None
    with open(filepath) as fi:
        lines = fi.readlines()

    for line in lines:
        tokens = line.split(",")
        _id, imagefile = int(tokens[0]), tokens[-1].strip().replace("\n", "")
        if wine is None:
            wine = _id
        elif wine != _id:
            shuffle(data, labels)
            size = len(labels)
            tr_end = 1 * size//4
            t_end = tr_end + (size - tr_end) // 2
            dataset["train"]["x"] += data[:tr_end]
            dataset["train"]["y"] += labels[:tr_end]
            dataset["test"]["x"] += data[tr_end:t_end]
            dataset["test"]["y"] += labels[tr_end:t_end]
            dataset["validation"]["x"] += data[t_end:]
            dataset["validation"]["y"] += labels[t_end:]
            wine = _id
            data = []
            labels = []
        img = Image.open(os.path.join(dir_path, imagefile))
        data.append(to_np(img, dim))
        labels.append(wine)

    return dataset

def to_np(image, dim):
    arr = []
    w, h = image.size
    for x in range(w):
        sub_array = []
        for y in range(h):
            sub_array.append(image.load()[x, y][:3])
        arr.append(sub_array)
    image_data = np.array(arr)
    image_data = np.array(np.reshape(image_data, (dim[0], dim[1], 3)))
    return image_data

def np_save(x, y, dir_path):
    #x = np.array(x)
    y = np.array(y)
    np.save('{}/x.npy'.format(dir_path), x)
    np.save('{}/y.npy'.format(dir_path), y)

def shuffle(X,Y):
    #shuffle examples and labels arrays together
    rng_state = np.random.get_state()
    np.random.shuffle(X)
    np.random.set_state(rng_state)
    np.random.shuffle(Y)

def main():
    in_filepath = sys.argv[1]
    out_dir = sys.argv[2]
    dataset = import_images(in_filepath, DIMENSION)
    print(dataset["validation"]['x'])
    print(dataset["validation"]['y'])
    np_save(dataset["train"]['x'], dataset["train"]['y'], "%s/train" % (out_dir))
    np_save(dataset["test"]['x'], dataset["test"]['y'], "%s/test" % (out_dir))
    np_save(dataset["validation"]['x'], dataset["validation"]['y'], "%s/validation" % (out_dir))

if __name__ == '__main__':
    main()
