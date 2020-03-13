import sys
import traceback
from PIL import Image

def crop_image(image):
    """
    Args:
        image(Image): image
    Returns:
        Image: cropped image
    """
    cropped = [] 
    width, height = image.size
    print(width, height)
    ratio = float(240) / float(width)
    print("ratio", ratio)
    #width = width * ratio
    width = 240
    height = int(height * ratio)
    print("height", height)
    im = image.resize((width, height))
    cropped.append(im)
    h = 600
    diff = (height - h) // 2 
    for i in range(3):
        upper = i * diff 
        box = (0, upper, width, height - (2 * diff - upper))
        try:
            cropped.append(im.crop(box))
        except:
            traceback.print_exc(file=sys.stdout)    
    return cropped

def rotate_image(image):
    images = []
    images.append(image)
    images.append(image.rotate(90))
    images.append(image.rotate(180))
    images.append(image.rotate(270))
    return images

def main():
    filename = sys.argv[1]
    pathname = filename.split("/")
    parts = pathname[-1].split(".")
    im = Image.open(filename)
    images = crop_image(im)
    for i, image in enumerate(images):
        for j, x in enumerate(rotate_image(image)):
            x.save("%s_%d_%d.%s" % (parts[0], i, j, parts[-1]))

if __name__ == '__main__':
    main()

