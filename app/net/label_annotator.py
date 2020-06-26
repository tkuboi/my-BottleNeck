import sys
import os
import traceback
from PIL import Image
from image_utils import crop_image

DIM = (240, 320)
RESIZED = (30, 40)

def read_directories(dir_path, out_dir):
    count = 0
    for item in os.listdir(dir_path):
        path = os.path.join(dir_path, item)
        if os.path.isdir(path):
            wine_id = item.split("_")[0]
            images, coords, file_names = read_directory(path)
            save_coordinates(wine_id, file_names, coords, out_dir)
            count += save_images(wine_id, images, coords, out_dir)
    return count

def read_directory(dir_path):
    label0 = Image.open(dir_path + "/label.png")
    labels = read_label(label0, RESIZED[0]/DIM[0])
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
            score_grids = slide_filters(small, labels)
            #sorted_grids = [_sort(g) for g in score_grids]
            mins = get_mins(score_grids, labels)
            mindex, min_val = get_min(mins)
            ims, cors, names = gen_images(
                [min_val], im, small, labels[mindex], item)
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

def gen_images(mins, im, small, label, file_name):
    images = []
    coords = []
    file_names = []
    for i, min_val in enumerate(mins):
        top_x, top_y, bot_x, bot_y = scale_up(
            im.size[0] / small.size[0],
            (min_val[0] % small.size[0], min_val[0] // small.size[0]),
            label[1],
            label[0].size[1])
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
    resized.append(
        (im.resize((int(small.size[0] * down_scale), int(small.size[1] * down_scale))),
        fraction, down_scale))
    down_scale = 4/5
    resized.append(
        (im.resize((int(small.size[0] * down_scale), int(small.size[1] * down_scale))),
        fraction, down_scale))
    return resized

def slide_filters(im, labels):
    results = []
    for label, _, fraction in labels:
        results.append(slide_filter(im, label, 1/fraction))
    return results

def slide_filter(im, label, scale):
    results = []
    for r in range(im.size[1]):
        for c in range(im.size[0]):
            results.append(_filter(im, c, r, label, scale))
    return results

def _filter(im, x, y, label, scale):
    score = 0 
    if x + label.size[0]  > im.size[0]\
        or y + label.size[1] > im.size[1]:
        return sys.maxsize 
    for r in range(label.size[1]):
        for c in range(label.size[0]):
            score += get_diff(label.getpixel((c, r)), im.getpixel((x + c, y + r))) * scale
    return score

def get_diff(pix1, pix2):
    return abs(pix1[0] - pix2[0]) + abs(pix1[1] - pix2[1]) + abs(pix1[2] - pix2[2])

def get_mins(grids, labels):
    results = []
    for i, grid in enumerate(grids):
        results.append(get_mindex(grid, 1 / labels[i][2]))
    return results

def get_min(results):
    min_pos_val = results[0]
    min_label_idx = 0 
    for i, pos_val in enumerate(results):
        if pos_val[1] < min_pos_val[1]:
            min_pos_val = pos_val
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
    if len(sys.argv) < 3:
        print("Usage: <in_dir> <out_dir>")
        exit()
    in_dir = sys.argv[1]
    out_dir = sys.argv[2]
    count = read_directories(in_dir, out_dir)
    print(count)

if __name__ == '__main__':
    main()

