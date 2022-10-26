import sys
import time
import random
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
    
    muted_array_blur = (blurred_array * mute_factor).astype(np.uint8)
    img_arr[ranges[0][0]:ranges[0][1], ranges[1][0]:ranges[1][1]] = muted_array_blur


def choose_norm_point(img_arr, max_val):
    max_val = np.amax(img_arr)
    max_points = np.nonzero(img_arr >= max_val)
    point_choice = random.randrange(len(max_points[0]))
    return np.transpose(max_points)[point_choice]


def choose_point_past_cutoff(img_arr, point_candidate_start=0):
    # slice the considered array to only the points-worthy candidates
    points_candidates = img_arr[point_candidate_start:, :]
    max_val = np.amax(points_candidates)
    max_points = np.nonzero(points_candidates >= max_val)
    point_choice = random.randrange(len(max_points[0]))
    return np.transpose(max_points)[point_choice] + [point_candidate_start, 0]


def choose_biased_points(points, num_selections):
    points = np.array(points, dtype=np.int64)
    num_points = points.shape[0]
    # TODO: need a cleaner way to bias more heavily to the left
    rem_points_prob = np.array([(1 / num_points * i) for i in range(num_points)], dtype=np.float64)
    norm_rem_points_prob = rem_points_prob / rem_points_prob.sum()

    indx = np.random.choice(num_points, replace=False, size=num_selections, p=norm_rem_points_prob)
    return points[indx]

def get_entropy_points(img_arr, num_points=10000, blur_radius=15, gaus_sigma=5, mute_factor=1, show_blur_img=False, point_candidate_start=300):
    entropy_img_arr = np.copy(img_arr)
    point_candidates = set()

    print("hello")
    start = time.time()
    i = 0
    while len(point_candidates) != num_points * 3:
        i += 1

        # pick a random point, and then do a gaussian blur around some radius of that point
        # going to require getting a snippet of the image?
        point = choose_point_past_cutoff(entropy_img_arr, point_candidate_start=point_candidate_start)
        if point is None:
            print("Only found {} eligible candidates".format(i+1))
            break

        bias_image_area_around_point(
            entropy_img_arr, point, radius=blur_radius,
            gaus_sigma=gaus_sigma, mute_factor=mute_factor
        )
        # lets us make sure that the points are unique via sets
        point_candidates.add(tuple(point))

        # making sure we don't get significant runtime
        if i > num_points * 6:
            raise ValueError("We're running rampant! Exiting at {} iterations of {} points, {} found".format(
                i, num_points, len(point_candidates)))

    points = choose_biased_points(list(point_candidates), num_selections=num_points)

    end = time.time()
    total = end - start
    print("Took {} seconds & {} loops for {} points | {} picked\n  - average {} seconds".format(
        total, i, len(point_candidates), len(points), total / len(points)
    ))

    if show_blur_img:
        entropy_img = Image.fromarray(np.fliplr(np.rot90(entropy_img_arr, axes=(-1, 0))))
        entropy_img.show()

    return np.array(list(points), dtype=np.int64)

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
