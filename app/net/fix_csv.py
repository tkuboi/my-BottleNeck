import sys

def read_file(filename):
    with open(filename) as inf:
        return inf.readlines()

def write_file(lines, outfile):
    with open(outfile, 'w') as of:
        for line in lines:
            of.write('%s,%s,%s,%s,%s\n' % (line[0], line[1], line[2], line[3], line[4]))

def process_lines(lines):
    new_lines = []
    for line in lines:
        tokens = line.split(",")
        filename = tokens[-1].replace("\n", "")
        parts = filename.split(".")
        for i in range(3):
            new_line = tokens[:-1]
            new_line.append("%s_%d.%s" % (parts[0],i,parts[1]))
            new_lines.append(new_line)
    return new_lines

def main():
    if len(sys.argv) < 3:
        print("USAGE: <infile> <outfile>")
        exit()
    infile = sys.argv[1]
    outfile = sys.argv[2]
    lines = read_file(infile)
    new_lines = process_lines(lines)
    write_file(new_lines, outfile)

if __name__ == '__main__':
    main()
