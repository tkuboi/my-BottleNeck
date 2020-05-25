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

def create_labels(label, images):
    labels = []
    filename = "%s_%d.%s"
    nameparts = label[-1].split(".")
    for i,im in enumerate(images):
        labels.append(label[:-1] + [filename % (nameparts[0], i, nameparts[1])])
    return labels

def read_images(dir_path, out_dir, id_wine_dict, dim, _id=None):
    files = os.listdir(dir_path)
    labels = []
    wines = set() 
    count = 0
    for i, item in enumerate(files):
        file_path = os.path.join(dir_path , item)
        print(file_path)
        if os.path.isdir(file_path):
            filename_ext = item.split('.')
            filename_parts = item.split('_')
            _id = int(filename_parts[0])
            count_labels = read_images(file_path, out_dir, id_wine_dict, dim, _id)
            count += count_labels[0]
            labels.extend(count_labels[1])
            wines.update(count_labels[2])
            continue
        try:
            filename_ext = item.split('.')
            filename_parts = item.split('_')
            image = Image.open(file_path)
            width, height = image.size
            print(width, height)
            if width > height:
                image = image.rotate(90)
            label = id_wine_dict[_id]
            #croppeds = generate_cropped_images(image, width, 1)
            rotated = [image]
            rotated.append(image.rotate(180))
            for j, cropped in enumerate(rotated):
                name = "%s_%d.%s" % (filename_ext[0], j+1, filename_ext[1]) 
                #resized = cropped.resize(dim)
                resized = cropped
                resized.save("%s/%s" % (out_dir, name))
                labels.append(label[:-1] + [name])
                count += 1
            wines.add(_id)
        except:
            traceback.print_exc(file=sys.stdout)
            print('WARNING : File {} could not be processed.'.format(file_path))
    return count, labels, wines

def add_trainer_images(bottles, in_dir, out_dir, wines, csvfile):
    count = 0
    labels = []
    processed = {}
    with open(csvfile) as wine_list:
        for wine in wine_list:
            tokens = wine.split(",")
            filename = tokens[-1]
            filename = filename.replace('\n', '').strip()
            _id = int(tokens[0])
            if _id not in wines:
                continue
            try:
                in_name = "%s/%s" % (in_dir, filename)
                image = Image.open(in_name)
                out_name = "%s/%s" % (out_dir, filename)
                image.save(out_name)
                labels.append(tokens[:-1] + [filename])
                count += 1
            except:
                traceback.print_exc(file=sys.stdout)
                print('WARNING : File {} could not be processed.'.format(filename))
    return count, labels

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
    #bgfilename = sys.argv[3]
    trainer_dir = sys.argv[3]

    img_labels, id_wine_dict = create_label_dict(trainer_dir+'/wineDataToImageFilename_2020_02_10.csv')
    count, labels, wines = read_images(in_dir, out_dir, id_wine_dict, DIMENSION)
    #count1, labels1 = add_trainer_images(labels, trainer_dir, out_dir, wines, trainer_dir+'/wineDataToImageFilename_2020_02_10.csv')
    #count += count1
    #labels += labels1
    write_labels(labels, out_dir + "/wineDataToImageFilename_2020_02_10.csv")

if __name__ == '__main__':
    main()
