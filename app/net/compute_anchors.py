from sklearn.cluster import KMeans
import numpy as np
import os
import sys
import argparse
import traceback

def read_directory(dir_path):
    boxes = []
    for item in os.listdir(dir_path):
        path = os.path.join(dir_path, item)
        if os.path.isdir(path):
            boxes += read_directory(path)
        elif ".txt" in item:
            try:
                tokens = item.split(".")
                name = tokens[0]
                boxes += get_boxes(path)
            except:
                print(path)
                traceback.print_exc(file=sys.stdout)
    return boxes

def get_boxes(path):
    boxes = []
    with open(path) as lines:
        for line in lines:
            tokens = line.split()
            boxes.append([float(tokens[3]), float(tokens[4])])
    return boxes[:1]

def get_anchors(boxes, num_anchors):
    x = np.array(boxes)
    kmeans = KMeans(n_clusters=num_anchors, random_state=0).fit(x)
    return kmeans.cluster_centers_

def save_results(anchors, file_path):
    string = ", ".join(["{}, {}".format(round(anchor[0], 5), round(anchor[1], 5))
                        for anchor in anchors])
    with open(file_path, 'w') as f:
        f.write(string)

def main(args):
    dir_path = args.directory_path
    if dir_path is None:
        print("Error: the directory path not given.")
        exit()
    num_anchors = args.num_anchors
    is_save = args.save_results
    boxes = read_directory(dir_path)
    anchors = get_anchors(boxes, num_anchors)
    print("Anchors:")
    print(anchors)
    if is_save:
        file_path = os.path.join(dir_path, 'anchors.txt')
        save_results(anchors, file_path)

if __name__ == '__main__':
    argparser = argparse.ArgumentParser(
        description='Compute anchor boxes for YOLO.')

    argparser.add_argument(
        '-d',
        '--directory_path',
        help='path to a directory containing text files of bounding boxes')

    argparser.add_argument(
        '-n',
        '--num_anchors',
        type=int,
        help='the number of anchor boxes to generate.',
        default=5)

    argparser.add_argument(
        '-s',
        '--save_results',
        help='save the results to a file in the directory.',
        action='store_true',
        default=False)

    args = argparser.parse_args()
    main(args)
