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
            _id = int(dir_path.split("_")[0]) 
            img = Image.open(path)
            if img.size[0] > img.size[1]:
                img = img.rotate(-90, expand=1)
            data.append(to_np(img.resize(dim), dim))
            label.append(_id)
            img.close()
            count += 1
        except:
            traceback.print_exc(file=sys.stdout)
    if len(data) > 0:
        file_x = "%s/%d_x.npy" % (out_dir, label[-1])
        file_y = "%s/%d_y.npy" % (out_dir, label[-1])
        np_save(data, label, file_x, file_y)
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

def np_save(x, y, out_file_x, out_file_y):
    #x = np.array(x)
    y = np.array(y)
    np.save(out_file_x, x)
    np.save(out_file_y, y)

def main():
    if len(sys.argv) < 3:
        print("Usage: <in_dir> <out_dir>")
        exit()
    in_filepath = sys.argv[1]
    out_dir = sys.argv[2]
    label_dict = import_images(in_filepath, DIMENSION, out_dir)
    
if __name__ == '__main__':
    main()
