import sys
import os
import traceback
import numpy as np
from PIL import Image
from image_utils import crop_image
from tripletloss_test import get_dists
from tensorflow.keras.models import load_model
from utils import to_np

DIM = (240, 320)
#RESIZED = (30, 40)
RESIZED = (240, 320)

def read_directories(dir_path, out_dir, model):
    count = 0
    for item in os.listdir(dir_path):
        path = os.path.join(dir_path, item)
        if os.path.isdir(path):
            wine_id = item.split("_")[0]
            images, coords, file_names = read_directory(path, model)
            save_coordinates(wine_id, file_names, coords, out_dir)
            count += save_images(wine_id, images, coords, out_dir)
    return count

def read_directory(dir_path, model):
    label0 = Image.open(dir_path + "/label.png")
    labels = read_label(label0, RESIZED[0]/DIM[0])
    label_arr = [to_np(label[0].resize((150, 200)), (150, 200, 3)) for label in labels]
    label_vecs = model.predict(np.array(label_arr))
    images = []
    coords = []
    file_names = []
    for item in os.listdir(dir_path):
        path = os.path.join(dir_path, item)
        if os.path.isdir(path) or "label" in item:
            continue
        try:
            im = Image.open(path)
            if im.size[0] > im.size[1]:
                im = im.rotate(-90, expand=1)
            small = im.resize(RESIZED)
            #im_array = np.array(small)
            slice_pos_list = slide_filters(small, labels)
            #print("slice_list", slice_list)
            score_grids = get_scores(model, slice_pos_list, label_vecs)
            #sorted_grids = [_sort(g) for g in score_grids]
            mins = get_mins(score_grids)
            mindex, min_pos_val = get_min(mins, labels)
            print(mindex, min_pos_val, len(score_grids))
            min_pos = score_grids[mindex][1][min_pos_val[0]]
            ims, cors, names = gen_images(
                [min_pos], im, small, labels[mindex], item)
            images += ims
            coords += cors
            file_names += names
            #for i, grid in enumerate(sorted_grids):
            #    ims, cors, names = gen_images(grid[:1], im, small, label0, labels[i], item)
            #    images += ims
            #    coords += cors
            #    file_names += names
        except:
            traceback.print_exc(file=sys.stdout)
    return images, coords, file_names

def get_scores(model, slice_pos_list, label_vecs):
    score_grids = []
    for (slices, positions), label in zip(slice_pos_list, label_vecs):
        print(slices)
        embeds = model.predict(slices)
        scores = get_dists(embeds, label)
        score_grids.append((scores, positions))
    return score_grids

def gen_images(mins, im, small, label, file_name):
    images = []
    coords = []
    file_names = []
    for i, min_val in enumerate(mins):
        """top_x, top_y, bot_x, bot_y = scale_up(
            im.size[0] / small.size[0],
            (min_val[0], min_val[1]),
            label[1],
            label[0].size[1])"""
        top_x, top_y = min_val[0], min_val[1]
        bot_x, bot_y = top_x + label[0].size[0], top_y + label[0].size[1]
        cropped = crop_image(im, (top_x, top_y, bot_x, bot_y))
        images.append(cropped)
        coords.append((top_x, top_y, bot_x, bot_y))
        file_names.append(file_name)
    return images, coords, file_names

def scale_up(scale1, pos, fraction, ah):
    top_x = int(pos[0] * scale1)
    top_y = int(pos[1] * scale1)
    h = int(ah / fraction)
    print(top_x, top_y, h)
    h = h - h % 4 
    w = int(3 * h / 4)

    bot_x = top_x + w
    bot_y = top_y + h
    return top_x, top_y, bot_x, bot_y 

def read_label(im, fraction):
    #im = Image.open(dir_path + "/label.png")
    resized = []
    # 30 x 40
    #resized.append(im)
    small = im.resize((int(im.size[0] * fraction), int(im.size[1] * fraction)))
    resized.append((small, fraction, 1.0))
    down_scale = 2/3
    w = int(small.size[0] * down_scale / 3)
    resized.append(
        (im.resize((w * 3, w * 4)),
        fraction, down_scale))
    down_scale = 4/5
    w = int(small.size[0] * down_scale / 3)
    resized.append(
        (im.resize((w * 3, w * 4)),
        fraction, down_scale))
    up_scale = 5/4
    w = int(small.size[0] * up_scale / 3)
    resized.append(
        (im.resize((w * 3, w * 4)),
        fraction, up_scale))
    return resized

def slide_filters(im, labels):
    results = []
    for label, _, fraction in labels:
        result = slide_filter(im, label, 1/fraction)
        if len(result[0]) > 0:
            results.append((np.array(result[0]), result[1]))
    return results

def slide_filter(im, label, scale):
    results = []
    positions = []
    for r in range(0, im.size[1], 4):
        for c in range(0, im.size[0], 4):
            if c + label.size[0]  > im.size[0]\
                or r + label.size[1] > im.size[1]:
                    continue
            results.append(_filter(im, c, r, label))
            positions.append((c, r))
    return results, positions

def _filter(im, x, y, label):
    return to_np(
            im.crop((x, y, label.size[0], label.size[1])).resize((150, 200)),
            (150, 200, 3))

def get_diff(pix1, pix2):
    return abs(pix1[0] - pix2[0]) + abs(pix1[1] - pix2[1]) + abs(pix1[2] - pix2[2])

def get_mins(grids):
    results = []
    for i, grid in enumerate(grids):
        results.append(get_mindex(grid[0]))
    return results

def get_min(results, labels):
    min_pos_val = results[0]
    min_val = results[0][1] / labels[0][2]
    min_label_idx = 0 
    for i, pos_val in enumerate(results):
        if pos_val[1] / labels[i][2]  < min_val:
            min_pos_val = pos_val
            min_val = pos_val[1] / labels[i][2]
            min_label_idx = i
    return min_label_idx, min_pos_val

def get_mindex(grid, scale=1.0):
    min_val = grid[0]
    mindex = 0
    for i, score in enumerate(grid):
        if score < min_val:
            min_val = score
            mindex = i
    return mindex, min_val * scale

def _sort(scores):
    scores = [(i, s) for i, s in enumerate(scores)]
    return sorted(scores, key=lambda x: x[1])

def save_images(wine_id, images, coords, out_dir):
    count = 0
    filename = "%s/%s/id_%s_label%d.png" 
    for i, image in enumerate(images):
        image.save(filename % (out_dir, wine_id, wine_id, i,))
        count += 1
    return count

def save_coordinates(wine_id, file_names, coords, out_dir):
    dir_path = "%s/%s" % (out_dir, wine_id)
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)
    file_path = "%s/%s/id_%s_coordinates.tsv" % (out_dir, wine_id, wine_id)
    with open(file_path, 'w') as of:
        for i, coord in enumerate(coords):
            of.write("%s\t%s\t%s\t%s\t%s\n" % (
                file_names[i], coord[0], coord[1], coord[2], coord[3]))

def main():
    if len(sys.argv) < 5:
        print("Usage: <in_dir> <out_dir> <model_path> <weight_path>")
        exit()
    in_dir = sys.argv[1]
    out_dir = sys.argv[2]
    model_path = sys.argv[3]
    weight_path = sys.argv[4]
    model = load_model(model_path)
    model.load_weights(weight_path)
    count = read_directories(in_dir, out_dir, model)
    print(count)

if __name__ == '__main__':
    main()

