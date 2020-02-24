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
    cropped = None
    width, height = image.size
    bottom = 2 * height // 3
    box = (0, bottom, width, height)
    try:
        cropped = image.crop(box)
    except:
        traceback.print_exc(file=sys.stdout)    
    return cropped

def main():
    filename = sys.argv[1]
    im = Image.open(filename)
    im = crop_image(im)
    im.show()

if __name__ == '__main__':
    main()

