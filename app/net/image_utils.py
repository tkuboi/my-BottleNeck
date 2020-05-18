import sys
import traceback
from PIL import Image

def extract_filename(path):
    parts = path.split("/")
    return parts[-1]

def remove_transparent(image, val=0):
    width, height = image.size
    for w in range(width):
        for h in range(height):
            pix = image.getpixel((w, h))
            if len(pix) > 3 and pix[3] < 1:
                image.putpixel((w, h), (255, 255, 255, 1))
    return image

def scale_image(image, dim=240, down_scale=True):
    """
    Args:
        image(Image): image
        dim (int): dimension
    Returns:
        Image: scaled image
    """
    width, height = image.size
    #print(width, height)
    if height >= width:
        if down_scale:
            ratio = float(dim) / float(height)
            height = dim
            width = int(width * ratio)
        else:
            ratio = float(dim) / float(width)
            width = dim 
            height = int(height * ratio)
    else:
        if down_scale:
            ratio = float(dim) / float(width)
            width = dim 
            height = int(height * ratio)
        else:
            ratio = float(dim) / float(height)
            height = dim 
            width = int(width * ratio)
    return image.resize((width, height))

def crop_image(image, box):
    #print(box)
    return image.crop(box)

def generate_cropped_images(image, dim, offset=0):
    cropped = [] 
    width, height = image.size
    diff = (height - dim) // 2 
    for i in range(offset, 3):
        upper = i * diff 
        box = (0, upper, width, upper + dim)
        try:
            cropped.append(crop_image(image, box)) 
        except:
            traceback.print_exc(file=sys.stdout)
    image.close()
    return cropped

def paste_image(image, background, box=None):
    base = background.copy()
    w, h = image.size
    #print("image.size",image.size)
    #print("base.size",base.size)
    if box is None:
        box = ((base.size[0] - w) // 2, 0)
    #print(box)
    base.paste(image, box)
    return base 

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

