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


def bias_image_area_around_point(img_arr, point, width=5, gaus_sigma=5, mute_factor=1):
    # set up the numpy array for the default range
    ranges = np.asarray(
        [[-width, width+1],
         [-width, width+1]]
    )

    # add in the point indexes
    ranges[0] += point[0]
    ranges[1] += point[1]

    # clean the ranges minimum to 0, negatives act weird
    ranges[ranges < 0] = 0

    # get the subarray (numpy indexing auto-respects ranges higher than widths/heights)
    blurred_array = gaussian_filter(
        img_arr[ranges[0][0]:ranges[0][1], ranges[1][0]:ranges[1][1]], sigma=gaus_sigma)

    img_arr[ranges[0][0]:ranges[0][1], ranges[1][0]:ranges[1][1]] = (blurred_array * mute_factor).astype(np.uint8)


def get_entropy_points(entropy_img_arr, num_points=10000, blur_radius=15, gaus_sigma=5, mute_factor=1):
    points = []

    start = time.time()
    print("hello")
    for _ in range(num_points):
        max_val = np.amax(entropy_img_arr)
        max_points = np.nonzero(entropy_img_arr >= max_val)

        # pick a random point, and then do a gaussian blur around some radius of that point
        # going to require getting a snippet of the image?
        point_choice = randrange(len(max_points[0]))
        point = max_points[0][point_choice], max_points[1][point_choice]
        points.append(point)
        bias_image_area_around_point(
            entropy_img_arr, point, width=blur_radius,
            gaus_sigma=gaus_sigma, mute_factor=mute_factor
        )

    end = time.time()
    total = end - start
    print("Took {} seconds for {} points\n  - average {} seconds".format(
        total, num_points, total / num_points
    ))

    return np.array(points)

def draw_points_as_delany_triangles(img_arr, img_points):
    tri = Delaunay(img_points)
    img = Image.fromarray(img_arr)
    draw = ImageDraw.Draw(img)

    for tri_indexes in tri.simplices:
        points = [tuple(tri.points[x]) for x in tri_indexes]
        draw.polygon(points, outline="#ffffff")
    
    return img

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
