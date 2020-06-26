import os
import sys
import traceback
import numpy as np
from PIL import Image

def import_images(dir_path, dim):
    data =  []
    label = []
    count = 0
    for item in os.listdir(dir_path):
        path = os.path.join(dir_path, item)
        if os.path.isdir(path):
            x, y = import_images(path, dim)
            data += x
            label += y
            continue
        try:
            _id = int(item.split("_")[1])
            img = Image.open(path)
            data.append(to_np(img.resize(dim), dim))
            label.append(_id)
            img.close()
            count += 1
        except:
            traceback.print_exc(file=sys.stdout)
    return (data, label)

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

def np_save(x, y, out_file_x, out_file_y):
    #x = np.array(x)
    #y = np.array(y)
    np.save(out_file_x, x)
    np.save(out_file_y, y)

def shuffle(X,Y):
    #shuffle examples and labels arrays together
    rng_state = np.random.get_state()
    np.random.shuffle(X)
    np.random.set_state(rng_state)
    np.random.shuffle(Y)
