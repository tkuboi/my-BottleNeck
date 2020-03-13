"""Data Preprocessing program

Author:
    Toshi Kuboi
"""

import numpy as np
import os
import sys
import traceback
from PIL import Image
from crop_image import crop_image

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
                filename_label[filename] = "%s %s" % (items[2], items[3])
                print(filename)
            line = csv.readline()
            i += 1
    return filename_label

def paste_image(image):
    w, h = image.size
    box = ((450 - w) // 2, 0)
    pasted = Image.open("resources/white_portlate.png")
    pasted.paste(image, box)
    return pasted
 
def read_images(dir_path, out_dir, img_labels):
    files = os.listdir( dir_path )
    labels = []
    count = 0
    for i, item in enumerate(files):
        file_path = os.path.join(dir_path , item)
        if os.path.isdir(file_path):
            continue
        try:
            image = Image.open(file_path)
            pasted = paste_image(image)
            pasted.save("%s/" % (out_dir) + item)
            cropped = crop_image(pasted)
            labels.append(img_labels[item.lower()])
            #resized_images = [image.resize((dim, dim)) for image in cropped]
            count += 1
        except:
            traceback.print_exc(file=sys.stdout)
            print('WARNING : File {} could not be processed.'.format(file_path))
    return count, labels

def write_labels(labels, outfile):
    with open(outfile, 'w') as of:
        for label in labels:
            of.write("%s, %s, %s, %s\n" % (label))

def main():
    if len(sys.argv) < 3:
        print("usage trainer.py <in_dir> <out_dir> [<dim>]")
        exit()
    in_dir = sys.argv[1]
    out_dir = sys.argv[2]

    img_labels = create_label_dict(in_dir+'/wineDataToImageFilename_2020_02_10.csv')
    count, labels = read_images(in_dir, out_dir, img_labels)
    #write_labels(labels, "preprocessed/wineDataToImageFilename_2020_02_10.csv")

if __name__ == '__main__':
    main()
