
import sys
import os
import traceback
import pickle

def import_csv(filepath, out_dir):
    labels = {}
    count = 0
    with open(filepath) as f:
        for line in f:
            try:
                tokens = line.split(",")
                if not tokens[0].isnumeric():
                    continue
                _id = int(tokens[0])
                wine = ", ".join(tokens[2:4])
                labels[_id] = [token.replace('"', '').strip() for token in reversed(tokens[2:4])] 
                count += 1
            except:
                traceback.print_exc(file=sys.stdout)
    return labels

def main():
    if len(sys.argv) < 3:
        print("Usage: <filepath> <out_dir>")
        exit()
    filepath = sys.argv[1]
    out_dir = sys.argv[2]
    labels = import_csv(filepath, out_dir)
    pickle.dump(labels, open("%s/label_dict.pkl" % (out_dir), 'wb'))

if __name__ == '__main__':
    main()
