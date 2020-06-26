import os
import sys
import traceback
import pickle
import numpy as np
from PIL import Image

#DIMENSION = (240, 320)
DIMENSION = (150, 200)

def import_images(dir_path, dim, out_dir):
    data =  []
    label = []
    count = 0
    for item in os.listdir(dir_path):
        path = os.path.join(dir_path, item)
        if os.path.isdir(path):
            import_images(path, dim, out_dir)
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
    if len(data) > 0:
        file_x = "%s/%d_x.npy" % (out_dir, label[-1])
        file_y = "%s/%d_y.npy" % (out_dir, label[-1])
        np_save2(data, label, file_x, file_y)
    return

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
    if len(sys.argv) < 3:
        print("Usage: <in_dir> <out_dir>")
        exit()
    in_filepath = sys.argv[1]
    out_dir = sys.argv[2]
    label_dict = import_images(in_filepath, DIMENSION, out_dir)
    
if __name__ == '__main__':
    main()
