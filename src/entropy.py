import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from skimage import filters, morphology

from src.utils.file_utils import get_image_arr, save_image_arr_to_path


def get_image_entropy_with_highbias(bw_img_arr, entropy_width=3, square_highbias=True, low_cutoff=50):
    # general entropy fetch
    entropy_arr = filters.rank.entropy(
        bw_img_arr, morphology.disk(entropy_width))

    # uses the entropy normal, and optionally some exponential math to high-bias the entropy pixels
    entropy_arr_unit = entropy_arr / max(entropy_arr.flatten())
    if square_highbias:
        entropy_arr_unit = entropy_arr_unit * entropy_arr_unit

    # raise 256 to the power of <pixel> for each pixel (range 0-1) -- should have an x^2 - 1 like curve to it
    base_arr = np.zeros(shape=entropy_arr.shape) + 256
    high_bias_arr = (np.power(base_arr, entropy_arr_unit) - 1)
    high_bias_arr[high_bias_arr < 0] = 0  # handle stray negatives

    return high_bias_arr


def get_lowband_failure_array(bw_img_arr, low_cutoff=50):
    # low-band failure, hard-setting
    bw_img_arr[bw_img_arr < low_cutoff] = 0

    return bw_img_arr


def get_pixel_weight_histogram(img_arr):
    # quick metric
    plt.hist(img_arr.astype(np.uint8).flatten(), density=False,
             bins=256)  # density=False would make counts
    plt.ylabel('Probability')
    plt.xlabel('Data')
    plt.show()


def get_image_entropy(bw_img_arr, entropy_width, entropy_threshold, data_visualization=False, show_minmax=False):
    # pulling the entropy of the image
    high_bias_arr = get_image_entropy_with_highbias(
        bw_img_arr, entropy_width=entropy_width, low_cutoff=entropy_threshold)

    # show minor debug hints
    if show_minmax:
        print("{} vs {}".format(min(high_bias_arr.flatten()),
              max(high_bias_arr.flatten())))

    # quick data visualization
    if data_visualization:
        get_pixel_weight_histogram(high_bias_arr)

    return high_bias_arr


def save_entropy_image(filepath, entropy_img_arr, entropy_width, entropy_threshold):
    save_dir = './images/entropy'
    filename = os.path.splitext(os.path.basename(filepath))[0]
    save_path = os.path.join(
        save_dir, '{}_{}-res_{}-cutoff.png'.format(filename, entropy_width, entropy_threshold))
    save_image_arr_to_path(entropy_img_arr, save_path)


if __name__ == "__main__":
    n = len(sys.argv)
    print("Total arguments passed: {}".format(n-1))
    if n < 2:
        raise ValueError("Please pass in the filename for the script to use")

    filepath = sys.argv[1]
    entropy_width = sys.argv[2] if n > 2 else 3
    entropy_threshold = sys.argv[3] if n > 3 else 50

    # grabbing & loading the image
    bw_img_arr = get_image_arr(filepath, greyscale=True)
    entropy_img_arr = get_image_entropy(
        bw_img_arr, entropy_width, entropy_threshold, data_visualization=True)

    # saving the image entropy
    save_entropy_image(filepath, entropy_img_arr,
                       entropy_width, entropy_threshold)
