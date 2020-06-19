import os
import sys
import traceback
import pickle
import weakref
import time
import numpy as np
import copy
from PIL import Image

DIMENSION = (240, 320)

def get_filelist(dir_path):
    filelist = {}
    files = os.listdir(dir_path)
    for item in files:
        file_path = os.path.join(dir_path, item)
        head, tail = os.path.splitext(item)
        if os.path.isdir(file_path) or tail != '.npy' or '_x' not in head:
            continue
        _id = item.split("_")[0]
        filelist[_id] = file_path
    return filelist

def create_dataset(file_list, batch, epoch, dim, num_wines=256):
    dataset = {"train":{"x":[], "y":[]}, "validation":{"x":[], "y":[]},
               "test":{"x":[], "y":[]}, "unique":{"x":[], "y":[]}}
    unique = set()
    keys = list(file_list.keys())
    np.random.shuffle(keys)
    step = batch // 2 
    for n in range(epoch):
        print("epoch=", n)
        for i in range(0, num_wines, batch):
            j = 0
            count = 0
            while count < batch:
                try:
                    _id = keys[i + j]
                    print(i, file_list[_id])
                    npx = np.load(file_list[_id])
                    #w = weakref.proxy(npx)
                    print(npx.shape)
                    if npx.shape[0] < 10:
                        j += 1
                        continue
                    #if n == 0:
                    #    np.random.shuffle(npx)
                    if _id not in unique:
                        dataset["unique"]["x"].append(copy.deepcopy(npx[0]))
                        dataset["unique"]["y"].append(_id)
                        unique.add(_id)
                    for k in range(2):
                        dataset["train"]["x"].append(copy.deepcopy(npx[k]))
                        dataset["train"]["y"].append(_id)
                        count += 1
                        if j % 2 == 1:
                            continue
                        dataset["validation"]["x"].append(copy.deepcopy(npx[k+3]))
                        dataset["validation"]["y"].append(_id)
                        dataset["test"]["x"].append(copy.deepcopy(npx[k+6]))
                        dataset["test"]["y"].append(_id)
                except:
                    traceback.print_exc(file=sys.stdout)
                finally:
                    del npx
                    j += 1
        #time.sleep(60)
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

def np_save2(x, y, out_file_x, out_file_y):
    #x = np.array(x)
    y = np.array(y)
    np.save(out_file_x, x)
    np.save(out_file_y, y)

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
    if len(sys.argv) < 6:
        print("Usage: <in_dir> <out_dir> <num_wines> <batch> <epoch>")
        exit()
    in_dir = sys.argv[1]
    out_dir = sys.argv[2]
    num_wines = int(sys.argv[3])
    batch = int(sys.argv[4])
    epoch = int(sys.argv[5])
    file_list = get_filelist(in_dir)
    dataset = create_dataset(file_list, batch, epoch, DIMENSION, num_wines)
    np_save(dataset["train"]['x'], dataset["train"]['y'], "%s/train" % (out_dir))
    np_save(dataset["validation"]['x'], dataset["validation"]['y'], "%s/validation" % (out_dir))
    np_save(dataset["test"]['x'], dataset["test"]['y'], "%s/test" % (out_dir))
    np_save(dataset["unique"]['x'], dataset["unique"]['y'], "%s/unique" % (out_dir))
    
if __name__ == '__main__':
    main()
