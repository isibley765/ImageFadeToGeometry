import os
from PIL import Image, ImageOps
import numpy as np


def get_image_arr(filepath, greyscale=False):
    # grabbing & loading the image
    img = Image.open(filepath)
    if greyscale:
        img = ImageOps.grayscale(img)
    
    return np.asarray(img)

def save_image_arr_to_path(img_arr, filepath):
    # get the image array to an Image object w/ a savable color mode
    img = Image.fromarray((img_arr).astype(np.uint8))
    save_image_to_path(img, filepath)

def save_image_to_path(img, filepath):
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    # make sure the folderpath exists
    save_dir = os.path.dirname(filepath)
    os.makedirs(save_dir, exist_ok=True)

    # save the file
    print("Saving to:\n  {}".format(filepath))
    img.save(filepath)
