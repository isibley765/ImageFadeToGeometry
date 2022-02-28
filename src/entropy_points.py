import sys
import time
from random import randrange
from typing import DefaultDict

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage.filters import gaussian_filter
from scipy.spatial import Delaunay

from src.entropy import get_image_entropy
from src.utils.file_utils import get_image_arr, save_image_arr_to_path
from src.utils.image_utils import find_square_around_point


def bias_image_area_around_point(img_arr, point, radius=5, gaus_sigma=5, mute_factor=1):
    # find the area around the point for the gaussian filter to use
    ranges = find_square_around_point(point, radius=radius)

    # get the subarray (numpy indexing auto-respects ranges higher than widths/heights)
    blurred_array = gaussian_filter(
        img_arr[ranges[0][0]:ranges[0][1], ranges[1][0]:ranges[1][1]], sigma=gaus_sigma)

    img_arr[ranges[0][0]:ranges[0][1], ranges[1][0]:ranges[1][1]] = (blurred_array * mute_factor).astype(np.uint8)


def get_entropy_points(img_arr, num_points=10000, blur_radius=15, gaus_sigma=5, mute_factor=1, show_blur_img=False):
    entropy_img_arr = np.copy(img_arr)
    points = set()

    print("hello")
    start = time.time()
    i = 0
    while len(points) != num_points:
        i += 1
        max_val = np.amax(entropy_img_arr)
        max_points = np.nonzero(entropy_img_arr >= max_val)

        # pick a random point, and then do a gaussian blur around some radius of that point
        # going to require getting a snippet of the image?
        point_choice = randrange(len(max_points[0]))
        point = max_points[0][point_choice], max_points[1][point_choice]
        bias_image_area_around_point(
            entropy_img_arr, point, radius=blur_radius,
            gaus_sigma=gaus_sigma, mute_factor=mute_factor
        )
        # lets us make sure that the points are unique... may involve significant runtime?
        point_str = ".".join(str(el) for el in point)
        points.add(point_str)

        # making sure we don't get significant runtime
        if i > num_points * 4:
            raise ValueError("We're running rampant! Exiting at {} iterations, only {}/{} points".format(
                i, len(points), num_points))

    end = time.time()
    total = end - start
    print("Took {} seconds & {} loops for {} points\n  - average {} seconds".format(
        total, i, num_points, total / num_points
    ))

    if show_blur_img:
        entropy_img = Image.fromarray(np.fliplr(np.rot90(entropy_img_arr, axes=(-1, 0))))
        entropy_img.show()

    return np.array([[int(el) for el in point.split('.')] for point in points])

if __name__ == "__main__":
    n = len(sys.argv)
    print("Total arguments passed: {}".format(n-1))
    if n < 2:
        raise ValueError("Please pass in the filename for the script to use")

    filepath = sys.argv[1]
    blur_radius = int(sys.argv[2]) if n > 2 else 5
    entropy_width = int(sys.argv[3]) if n > 3 else 3
    entropy_threshold = int(sys.argv[4]) if n > 4 else 50

    # grabbing & loading the image
    bw_img_arr = get_image_arr(filepath, greyscale=True)
    # getting the entropy of the image
    entropy_img_arr = get_image_entropy(
        bw_img_arr, entropy_width, entropy_threshold)

    # grab ya some points
    img_points = get_entropy_points(entropy_img_arr, blur_radius=blur_radius)
    points_inverted = img_points.T
    tri = Delaunay(img_points)

    import matplotlib.pyplot as plt
    plt.triplot(img_points[:, 0], img_points[:, 1], tri.simplices)
    plt.plot(img_points[:, 0], img_points[:, 1], 'o')
    plt.show()

    # debugging some stuff
    img_mask = np.zeros(shape=entropy_img_arr.shape)
    img_mask[points_inverted[0], points_inverted[1]] = 255

    save_image_arr_to_path(img_mask, './images/white_tiger_dot_child.png')
    entropy_img_arr[points_inverted[0], points_inverted[1]] = 255
    Image.fromarray(entropy_img_arr).show()
    print(img_points)
    print("oof")
