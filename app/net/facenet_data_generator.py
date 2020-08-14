import os
import sys
import traceback
import pickle
import weakref
import time
import numpy as np
import copy
from PIL import Image

DIMENSION = (150, 200)

def get_files_dict(dir_path):
    filelist = {}
    files = os.listdir(dir_path)
    for item in files:
        file_path = os.path.join(dir_path, item)
        head, tail = os.path.splitext(item)
        if os.path.isdir(file_path) or tail != '.npy' or '_x' not in head:
            continue
        _id = int(item.split("_")[0])
        filelist[_id] = file_path
    return filelist

def get_arrays_dict(files_dict):
    arrays_dict = {}
    for wine_id in files_dict.keys():
        arr = np.load(files_dict[wine_id])
        arrays_dict[wine_id] = arr
    return arrays_dict

def map_wine_winery(wines_dict):
    wine_winery_map = {}
    winerys = {}
    winery_id = 0
    for wine_id in wines_dict.keys():
        if wines_dict[wine_id][0] in winerys:
            wine_winery_map[wine_id] = winerys[wines_dict[wine_id][0]]
        else:
            winerys[wines_dict[wine_id][0]] = winery_id
            wine_winery_map[wine_id] = winery_id
            winery_id += 1
    return wine_winery_map

def get_generator(in_dir, wines_dict, batch_size=32):
    files_dict = get_files_dict(in_dir)
    arrays_dict = get_arrays_dict(files_dict)
    wine_winery_map = map_wine_winery(wines_dict)
    wine_ids = list(arrays_dict.keys())
    wine_idx_dict = {_id:0 for _id in wine_ids}
    wine_idx_list_dict = {_id:[*range(len(array))] for _id, array in arrays_dict.items()}

    def data_generator(winery=False):
        np.random.shuffle(wine_ids)
        wine_id_idx = 0
        while True:
            x = []
            y = []
            count = 0
            while count < batch_size:
                if wine_id_idx >= len(wine_ids):
                    wine_id_idx = 0
                    np.random.shuffle(wine_ids)
                wine_id = wine_ids[wine_id_idx]
                index = wine_idx_dict[wine_id]
                for i in range(4):
                    if index >= len(arrays_dict[wine_id]):
                        wine_idx_dict[wine_id] = index = 0
                    if index == 0:
                        np.random.shuffle(wine_idx_list_dict[wine_id])
                    array_idx = wine_idx_list_dict[wine_id][index]
                    x.append(arrays_dict[wine_id][array_idx])
                    if winery:
                        y.append(wine_winery_map[wine_id])
                    else:
                        y.append(wine_id)
                    index += 1
                    count += 1
                wine_id_idx += 1
            x = np.array(x).astype('float32')
            x /= 255.
            y = np.array(y).astype('int32')
            yield (x, y)

    return data_generator

def main():
    if len(sys.argv) < 6:
        print("Usage: <in_dir> <wines_pkl_path> <out_dir> <batch> <epoch>")
        exit()
    in_dir = sys.argv[1]
    wines_pkl_path = sys.argv[2]
    out_dir = sys.argv[3]
    batch = int(sys.argv[4])
    epoch = int(sys.argv[5])
    wines_dict = pickle.load(open(wines_pkl_path, 'rb'))
    #print(wines_dict)
    data_generator = get_generator(in_dir, wines_dict, batch)(True)
    X = []
    Y = []
    for i in range(epoch):
        x, y = next(data_generator)
        X += list(x)
        Y += list(y)
    X = np.array(X)
    Y = np.array(Y)
    print(X.shape)
    print(Y)
    np.save('{}/x.npy'.format(out_dir), X)
    np.save('{}/y.npy'.format(out_dir), Y)

if __name__ == '__main__':
    main()
