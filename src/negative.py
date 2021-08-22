import os
import sys

import numpy as np

from src.utils.file_utils import get_image_arr, save_image_arr_to_path


def get_negative(img_arr):
    return (255 - img_arr).astype(np.uint8)

if __name__ == "__main__":
    n = len(sys.argv)
    print("Total arguments passed:", n-1)
    if n < 2:
        raise ValueError("Please pass in the filename for the script to use")

    filepath = sys.argv[1]

    # grabbing & loading the image
    img_arr = get_image_arr(filepath)
    img_arr_negative = get_negative(img_arr)

    # inverting the image
    filename = os.path.splitext(os.path.basename(filepath))[0]
    save_image_arr_to_path(img_arr_negative, './images/negative/{}.png'.format(filename))
