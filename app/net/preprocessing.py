"""Data Preprocessing program

Author:
    Toshi Kuboi
"""

import numpy as np
import os
import sys
import traceback
from PIL import Image
from image_utils import crop_image, paste_image, scale_image, generate_cropped_images, remove_transparent

DIMENSION = 128

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

def read_images(dir_path, out_dir, img_labels, bgfilename):
    background = Image.open(bgfilename)
    dim = background.size[1]
    files = os.listdir( dir_path )
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
            resized = scale_image(image, dim)
            pasted = paste_image(resized, background)
            name = "%s/%s" % (out_dir, item) 
            pasted.save(name)
            labels.append(img_labels[item.lower()])
            count += 1
            #for j, cropped in enumerate(generate_cropped_images(image, dim)):
                #background = Image.open(bgfilename)
                #pasted = paste_image(cropped, background)
                #name = "%s/%s_%d.%s" % (out_dir, name_parts[0], j, name_parts[1]) 
                #cropped.save(name)
                #pasted.save(name)
                #background.close()
                #labels.append(img_labels[item.lower()])
                #count += 1
            #if count >= 10:
            #    break
        except:
            traceback.print_exc(file=sys.stdout)
            print('WARNING : File {} could not be processed.'.format(file_path))
    background.close()
    return count, labels

def read_images2(dir_path, out_dir, img_labels, dim):
    files = os.listdir( dir_path )
    labels = []
    count = 0
    for i, item in enumerate(files):
        file_path = os.path.join(dir_path , item)
        if os.path.isdir(file_path):
            continue
        try:
            name_parts = item.split('.')
            if name_parts[-2][-1] == '0':
                continue
            image = Image.open(file_path)
            width, height = image.size
            upper = (height - width) // 2
            box = (0, upper, width, upper + width)
            cropped = crop_image(image, box)
            resized = cropped.resize((dim, dim))
            name = "%s/%s" % (out_dir, item)
            resized.save(name)
            labels.append(img_labels[item.lower()])
            count += 1
        except:
            traceback.print_exc(file=sys.stdout)
            print('WARNING : File {} could not be processed.'.format(file_path))
    return count, labels

def read_images3(dir_path, out_dir, id_wine_dict, dim, _id=None):
    files = os.listdir(dir_path)
    labels = []
    count = 0
    for i, item in enumerate(files):
        file_path = os.path.join(dir_path , item)
        print(file_path)
        if os.path.isdir(file_path):
            filename_ext = item.split('.')
            filename_parts = item.split('_')
            _id = int(filename_parts[0])
            count_labels = read_images3(file_path, out_dir, id_wine_dict, dim, _id)
            count += count_labels[0]
            labels.extend(count_labels[1])
            continue
        try:
            filename_ext = item.split('.')
            filename_parts = item.split('_')
            image = Image.open(file_path)
            width, height = image.size
            label = id_wine_dict[_id]
            for j, cropped in enumerate(generate_cropped_images(image, width)):
                name = "%s_%d.%s" % (filename_ext[0], j+1, filename_ext[1]) 
                resized = cropped.resize((dim, dim))
                resized.save("%s/%s" % (out_dir, name))
                labels.append(label[:-1] + [name])
                count += 1
        except:
            traceback.print_exc(file=sys.stdout)
            print('WARNING : File {} could not be processed.'.format(file_path))
    return count, labels

def add_trainer_images(bottles, in_dir, out_dir, id_wine_dict):
    count = 0
    labels = []
    processed = {}
    for bottle in bottles:
        _id = int(bottle[0])
        if _id in processed:
            continue
        processed[_id] = True
        wine = id_wine_dict[_id]
        filename = wine[-1]
        file_ext = filename.split('.')
        nameparts = file_ext[0].split('_')
        basename = "_".join(nameparts[:-1])
        for i in range(1, 3):
            img_name = "%s_%d.%s" % (basename, i, file_ext[-1])
            in_name = "%s/%s" % (in_dir, img_name)
            image = Image.open(in_name)
            out_name = "%s/%s" % (out_dir, img_name)
            image.save(out_name)
            labels.append(wine[:-1] + [img_name])
            count += 1
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

    #img_labels, id_wine_dict = create_label_dict(in_dir+'/wineDataToImageFilename_2020_02_10.csv')
    img_labels, id_wine_dict = create_label_dict(trainer_dir+'/wineDataToImageFilename_2020_02_10.csv')
    #count, labels = read_images2(in_dir, out_dir, img_labels, DIMENSION)
    count, labels = read_images3(in_dir, out_dir, id_wine_dict, DIMENSION)
    count_labels = add_trainer_images(labels, trainer_dir, out_dir, id_wine_dict)
    count += count_labels[0]
    labels += count_labels[1]
    write_labels(labels, out_dir + "/wineDataToImageFilename_2020_02_10.csv")

if __name__ == '__main__':
    main()
