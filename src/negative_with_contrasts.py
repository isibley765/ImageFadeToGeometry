import os
import sys

import numpy as np

from src.negative import get_negative
from src.utils.file_utils import get_image_arr, save_image_arr_to_path


def get_image_with_contrast(img_arr, phi=1, theta=1, maxIntensity=255.0):
    # Parameters for manipulating image data
    phiArray = np.zeros(shape=(1, 1, 3)) + (maxIntensity/phi)
    thetaArray = np.zeros(shape=(1, 1, 3)) + (maxIntensity/theta)

    # tweaking the image
    img_arr_contrast = phiArray*(img_arr/thetaArray)**.5
    # normalize the vector to int units 0-255 again
    img_arr_contrast = (img_arr_contrast * (img_arr_contrast /
                        np.linalg.norm(img_arr_contrast.flatten(), ord=np.inf)))

    return (img_arr_contrast).astype(np.uint8)


if __name__ == "__main__":
    n = len(sys.argv)
    print("Total arguments passed:", n-1)
    if n < 2:
        raise ValueError("Please pass in the filename for the script to use")

    filepath = sys.argv[1]
    phi = float(sys.argv[2]) if n > 2 else 1
    theta = float(sys.argv[3]) if n > 3 else 1

    # grabbing & loading the image
    img_arr = get_image_arr(filepath)
    img_arr_contrast = get_image_with_contrast(img_arr, phi=phi, theta=theta)
    img_arr_contrast_negative = get_negative(img_arr_contrast)

    # save the results
    og_filename = os.path.splitext(os.path.basename(filepath))[0]
    filename = '{}_phi-{}_theta-{}.png'.format(og_filename, theta, phi)
    # save the version with just contrast
    save_image_arr_to_path(
        img_arr_contrast, './images/contrast/{}'.format(filename))
    # save the contrasted & inverted image
    save_image_arr_to_path(img_arr_contrast_negative,
                       './images/negative/contrast/{}'.format(filename))
