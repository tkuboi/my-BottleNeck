import sys
import os
import traceback
from PIL import Image

DIM = (240, 320)

def read_directory(dir_path, out_dir):
    count = 0
    for item in os.listdir(dir_path):
        path = os.path.join(dir_path, item)
        if os.path.isdir(path):
            count += read_directory(path, out_dir)
            continue
        try:
            im = Image.open(path)
            if im.size[0] > im.size[1]:
                im = im.rotate(-90, expand=1)
            im = im.resize(DIM)
            parts = path.split("/")
            folder = "/".join(parts[1:-1])
            out_dir_path = "%s/%s" % (out_dir, folder)
            if not os.path.exists(out_dir_path):
                os.mkdir(out_dir_path)
            out_path = "%s/%s" % (out_dir_path, item)
            im.save(out_path)
            count += 1
        except:
            traceback.print_exc(file=sys.stdout)
    return count

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
