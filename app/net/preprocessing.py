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
    with open(file_path) as csv:
        i = 0
        line = csv.readline()
        while line:
            if i != 0:
                items = line.split(',')
                filename = items[-1].split('/')[-1]
                filename = filename.replace('\n', '')
                filename_label[filename.lower()] = items[:-1] + [filename] 
                print(filename)
            line = csv.readline()
            i += 1
    return filename_label

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
    bgfilename = sys.argv[3]

    img_labels = create_label_dict(in_dir+'/wineDataToImageFilename_2020_02_10.csv')
    count, labels = read_images(in_dir, out_dir, img_labels, bgfilename)
    write_labels(labels, out_dir + "/wineDataToImageFilename_2020_02_10.csv")

if __name__ == '__main__':
    main()
