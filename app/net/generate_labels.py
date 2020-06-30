import sys
import os
import traceback
from PIL import Image

DIM = (150, 200)

def read_directory(dir_path, out_dir):
    count = 0
    for item in os.listdir(dir_path):
        path = os.path.join(dir_path, item)
        if os.path.isdir(path):
            count += read_directory(path, out_dir)
            continue
        if "label" not in item:
            continue
        try:
            im = Image.open(path)
            if im.size[0] > im.size[1]:
                im = im.rotate(-90, expand=1)
            im = im.resize(DIM)
            labels = [im]
            #labels = multi_flip(im)
            labels = multi_rotates(labels, 15)
            parts = path.split("/")
            folder = parts[-2]
            out_dir_path = "%s/%s" % (out_dir, folder)
            if not os.path.exists(out_dir_path):
                os.mkdir(out_dir_path)
            count += save_images(labels, out_dir_path, item)
        except:
            traceback.print_exc(file=sys.stdout)
    return count

def save_images(labels, out_dir_path, item):
    out_path = "%s/id_%s_label%d.%s"
    parts = item.split("_")
    wine_id = parts[1]
    ext = parts[-1].split(".")[-1]
    count = 0
    for i, label in enumerate(labels):
        label.save(out_path % (out_dir_path, wine_id, i, ext))
        count += 1
    return count

def multi_rotates(ims, angle=10):
    images = []
    for im in ims:
        images += multi_rotate(im, angle)
    return images

def multi_rotate(im, angle=10):
    images = []
    for d in range(-angle, angle):
        images.append(im.rotate(d, expand=1, fillcolor=(0, 0, 0)))
    return images

def multi_flip(im):
    images = [im]
    images.append(im.transpose(method=Image.FLIP_LEFT_RIGHT))
    return images

def main():
    if len(sys.argv) < 3:
        print("Usage: <in_dir> <out_dir>")
        exit()

    in_dir = sys.argv[1]
    out_dir = sys.argv[2]
    count = read_directory(in_dir, out_dir)
    print(count)

if __name__ == '__main__':
    main()
