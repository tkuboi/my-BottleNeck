"""Data Preprocessing program

Author:
    Toshi Kuboi
"""

import numpy as np
import os
import sys
import traceback
from PIL import Image
from image_utils import rotate_image, crop_image, paste_image, scale_image, generate_cropped_images, remove_transparent

DIMENSION = (240, 320) 

def save_images(images, out_dir, name):
    filepath = "%s/%s_%d.%s"
    nameparts = name.split(".")
    for i, im in enumerate(images):
        im.save(filepath % (out_dir, nameparts[0], i, nameparts[1]))

def create_label_dict(file_path):
    filename_label = dict()
    id_wine_dict = dict()
    with open(file_path) as csv:
        i = 0
        line = csv.readline()
        while line:
            if i != 0:
                items = line.split(',')
                filename = items[-1].split('/')[-1]
                filename = filename.replace('\n', '').strip()
                filename_label[filename.lower()] = items[:-1] + [filename]
                id_wine_dict[int(items[0])] = items[:-1] + [filename]
                print(filename)
            line = csv.readline()
            i += 1
    return filename_label, id_wine_dict

def read_images(dir_path, out_dir, img_labels, bgfiles, dim):
    files = os.listdir(dir_path)
    labels = []
    count = 0
    for i, item in enumerate(files):
        file_path = os.path.join(dir_path , item)
        if os.path.isdir(file_path):
            continue
        try:
            name_parts = item.split('.')
            image = Image.open(file_path)
            image = remove_transparent(image)
            if image.size[0] > image.size[1]:
                image = image.rotate(90)
            resized = scale_image(image, int(dim[1] * 1.20))
            w, h = resized.size
            box = (0, abs(h - dim[1]), w, h)
            cropped = crop_image(resized, box)
            pasteds = []
            name_ext = item.split(".")
            for background in bgfiles:
                for p in range((background.size[0] - cropped.size[0]) // 2 + 1):
                    box = (p, 0)
                    pasted = paste_image(cropped, background)
                    name = "%s_%d.%s" % (name_ext[0], p, name_ext[1])
                    pathname = "%s/%s" % (out_dir, name)
                    pasted.save(pathname)
                    pasteds.append(pasted)
                    labels.append(img_labels[item.lower()][:-1] + [name])
                    count += 1
        except:
            traceback.print_exc(file=sys.stdout)
            print('WARNING : File {} could not be processed.'.format(file_path))
    return count, labels

def get_bgfiles(dir_path):
    images = []
    files = os.listdir(dir_path)
    for i, item in enumerate(files):
        file_path = os.path.join(dir_path , item)
        if os.path.isdir(file_path):
            continue
        try:
            image = Image.open(file_path)
            images.append(image)
        except:
            traceback.print_exc(file=sys.stdout)
            print('WARNING : File {} could not be processed.'.format(file_path))
    return images

def close_images(images):
    for image in images:
        try:
            image.close()
        except:
            traceback.print_exc(file=sys.stdout)
            print('WARNING : File {} could not be processed.'.format(image.filename))

def create_labels(label, images):
    labels = []
    filename = "%s_%d.%s"
    nameparts = label[-1].split(".")
    for i,im in enumerate(images):
        labels.append(label[:-1] + [filename % (nameparts[0], i, nameparts[1])])
    return labels

def write_labels(labels, outfile):
    with open(outfile, 'w') as of:
        for label in labels:
            of.write("%s, %s, %s, %s, %s\n" % (tuple(label)))

def main():
    if len(sys.argv) < 4:
        print("usage preprocessing.py <in_dir> <out_dir> <background_filename>")
        exit()
    in_dir = sys.argv[1]
    out_dir = sys.argv[2]
    bgfile_dir = sys.argv[3]

    bgfiles = get_bgfiles(bgfile_dir)
    img_labels, id_wine_dict = create_label_dict(in_dir+'/wineDataToImageFilename_2020_02_10.csv')
    count, labels = read_images(in_dir, out_dir, img_labels, bgfiles, DIMENSION)
    write_labels(labels, out_dir + "/wineDataToImageFilename_2020_02_10.csv")
    close_images(bgfiles)

if __name__ == '__main__':
    main()
