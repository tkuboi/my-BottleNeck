import os
import sys
import traceback
import pickle
import numpy as np
from PIL import Image

DIMENSION = (240, 320)

def import_images(filepath, dim):
    dir_path = "/".join(filepath.split("/")[:-1])
    data = {} 
    labels = {}
    with open(filepath) as fi:
        lines = fi.readlines()
    
    count = 0
    for line in lines:
        tokens = line.split(",")
        wine = ", ".join(tokens[2:4])
        _id, imagefile = int(tokens[0]), tokens[-1].strip().replace("\n", "")
        if _id not in data:
            data[_id] = []
            count = 0
        if count > 29: 
            continue 
        img = Image.open(os.path.join(dir_path, imagefile))
        data[_id].append(to_np(img, dim))
        labels[_id] = wine
        img.close()
        count += 1
    return data, labels

def create_dataset(data_dict, batch, epoch):
    dataset = {"train":{"x":[], "y":[]}, "validation":{"x":[], "y":[]}}
    keys = data_dict.keys()
    np.random.shuffle(keys)
    step = batch // 2
    for n in range(epoch):
        for i in range(0, len(keys), step):
            j = 0
            while j < step:
                _id = keys[i + j]
                np.random.shuffle(data_dict[_id])
                for k in range(2):
                    dataset["train"]["x"].append(data_dict[_id][k])
                    dataset["train"]["y"].append(_id)
                    dataset["validation"]["x"].append(data_dict[_id][k+2])
                    dataset["validation"]["y"].append(_id)
                j += 1
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
    batch = int(sys.argv[3])
    epoch = int(sys.argv[4])
    data_dict, label_dict = import_images(in_filepath, DIMENSION)
    pickle.dump(label_dict, open("%s/label_dict.pkl" % (out_dir), 'wb'))
    del label_dict
    dataset = create_dataset(data_dict, batch, epoch)
    del data_dict
    np_save(dataset["train"]['x'], dataset["train"]['y'], "%s/train" % (out_dir))
    np_save(dataset["validation"]['x'], dataset["validation"]['y'], "%s/validation" % (out_dir))
    
if __name__ == '__main__':
    main()
