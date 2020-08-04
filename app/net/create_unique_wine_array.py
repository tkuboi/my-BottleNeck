import os
import sys
import traceback
import pickle
import weakref
import time
import numpy as np
import copy
from PIL import Image

from utils import to_np

DIMENSION = (150, 200)
IMAGE_EXTENSIONS = ['.png', '.jpg', '.JPG', '.jpeg']

def read_directory(dir_path, dim, num_per_wine=10):
    count = 0
    x = []
    y = []
    base_dir_name = os.path.basename(dir_path)
    prefix = base_dir_name.split("_")[0]
    wine_id = int(prefix) if prefix.isnumeric() else None
    items = os.listdir(dir_path)
    np.random.shuffle(items)
    for item in items:
        item_path = os.path.join(dir_path, item)
        if os.path.isdir(item_path):
            rx, ry = read_directory(item_path, dim, num_per_wine)
            x += rx
            y += ry
        elif wine_id:
            name, ext = os.path.splitext(item)
            if ext not in IMAGE_EXTENSIONS:
                continue
            try:
                img = Image.open(item_path)
                if img.size[0] > img.size[1]:
                    img = img.rotate(-90)
                img = img.resize(dim)
                image_data = to_np(img, dim)
                x.append(image_data)
                y.append(wine_id)
                print(wine_id)
                count += 1
            except:
                traceback.print_exc(file=sys.stdout) 
        if count > num_per_wine:
            break
    return x, y

def main():
    if len(sys.argv) < 3:
        print("Usage: <in_dir> <out_dir> <num_per_wine>")
        exit()
    in_dir_path = sys.argv[1]
    out_dir_path = sys.argv[2]
    num_per_wine = int(sys.argv[3]) if len(sys.argv) > 3 else 10 
    dim = DIMENSION
    x, y = read_directory(in_dir_path, dim, num_per_wine)
    x = np.array(x)
    y = np.array(y)
    np.save(os.path.join(out_dir_path, 'x.npy'), x)
    np.save(os.path.join(out_dir_path, 'y.npy'), y)
    print(y.shape[0])

if __name__ == '__main__':
    main()
