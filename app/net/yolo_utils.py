import os
import colorsys
import random

import numpy as np
from PIL import Image, ImageDraw, ImageFont

YOLO_ANCHORS = np.array(
    ((0.57273, 0.677385), (1.87446, 2.06253), (3.33843, 5.47434),
     (7.88282, 3.52778), (9.77052, 9.16828)))

def get_classes(classes_path):
    '''loads the classes'''
    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names

def get_anchors(anchors_path, is_proportional=False, image_wh=608):
    '''loads the anchors from a file'''
    if os.path.isfile(anchors_path):
        with open(anchors_path) as f:
            anchors = f.readline()
        if is_proportional:
            anchors = [float(x) * image_wh / 32 for x in anchors.split(',')]
        else:
            anchors = [float(x) for x in anchors.split(',')]
        return np.array(anchors).reshape(-1, 2)
    else:
        Warning("Could not open anchors file, using default.")
        return YOLO_ANCHORS

def resize_boundingboxes(out_boxes, image_size):
    boxes = []
    for out_box in out_boxes:
        top, left, bottom, right = out_box
        top = np.floor(top).astype('int32')
        left = np.floor(left).astype('int32')
        bottom = np.floor(bottom).astype('int32')
        right = np.floor(right).astype('int32')
        print(top, left, bottom, right)
        width = (right - left)
        height = (bottom - top)
        c_x = left + width // 2
        c_y = top + height // 2
        if width // 3  >= height // 4:
            width = int(width)
            height = width * 4 // 3
        else:
            height = int(height)
            width = height * 3 // 4
        top = max(0, c_y - height // 2)
        left = max(0, c_x - width // 2)
        bottom = min(image_size[1], c_y + height // 2)
        right = min(image_size[0], c_x + width // 2)
        box = (top, left, bottom, right)
        boxes.append(box)
    return boxes

def preprocess_image(image, model_image_shape):
    resized_image = image.resize(
            tuple(reversed(model_image_shape)), Image.BICUBIC)
    image_data = np.array(resized_image, dtype='float32')
    image_data = image_data[:, :, :3]
    image_data /= 255.
    image_input = np.expand_dims(image_data, 0)  # Add batch dimension.
    return image_input

def get_colors_for_classes(num_classes):
    """Return list of random colors for number of classes given."""
    # Use previously generated colors if num_classes is the same.
    if (hasattr(get_colors_for_classes, "colors") and
            len(get_colors_for_classes.colors) == num_classes):
        return get_colors_for_classes.colors

    hsv_tuples = [(x / num_classes, 1., 1.) for x in range(num_classes)]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(
        map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
            colors))
    random.seed(10101)  # Fixed seed for consistent colors across runs.
    random.shuffle(colors)  # Shuffle colors to decorrelate adjacent classes.
    random.seed(None)  # Reset seed to default.
    get_colors_for_classes.colors = colors  # Save colors for future calls.
    return colors


def draw_boxes(image, boxes, box_classes, class_names, scores=None, as_array=True):
    """Draw bounding boxes on image.
    Draw bounding boxes with class name and optional box score on image.
    Args:
        image: An `array` of shape (width, height, 3) with values in [0, 1].
        boxes: An `array` of shape (num_boxes, 4) containing box corners as
            (y_min, x_min, y_max, x_max).
        box_classes: A `list` of indicies into `class_names`.
        class_names: A `list` of `string` class names.
        `scores`: A `list` of scores for each box.
    Returns:
        A copy of `image` modified with given bounding boxes.
    """
    #image = Image.fromarray(np.floor(image * 255 + 0.5).astype('uint8'))
    base_dir = os.path.dirname(os.path.abspath(__file__))
    font = ImageFont.truetype(
        font=os.path.join(base_dir, 'font/FiraMono-Medium.otf'),
        size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
    thickness = (image.size[0] + image.size[1]) // 300

    colors = get_colors_for_classes(len(class_names))

    for i, c in list(enumerate(box_classes)):
        box_class = class_names[c]
        box = boxes[i]
        if isinstance(scores, np.ndarray):
            score = scores[i]
            label = '{} {:.2f}'.format(box_class, score)
        else:
            label = '{}'.format(box_class)

        draw = ImageDraw.Draw(image)
        label_size = draw.textsize(label, font)

        top, left, bottom, right = box
        top = max(0, np.floor(top + 0.5).astype('int32'))
        left = max(0, np.floor(left + 0.5).astype('int32'))
        bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
        right = min(image.size[0], np.floor(right + 0.5).astype('int32'))
        print(label, (left, top), (right, bottom))

        if top - label_size[1] >= 0:
            text_origin = np.array([left, top - label_size[1]])
        else:
            text_origin = np.array([left, top + 1])

        # My kingdom for a good redistributable image drawing library.
        for i in range(thickness):
            draw.rectangle(
                [left + i, top + i, right - i, bottom - i], outline=colors[c])
        draw.rectangle(
            [tuple(text_origin), tuple(text_origin + label_size)],
            fill=colors[c])
        draw.text(text_origin, label, fill=(0, 0, 0), font=font)
        del draw

    if as_array: 
        return np.array(image)
    return image
