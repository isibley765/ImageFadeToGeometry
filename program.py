import os
import sys

import numpy as np
from PIL import Image

from src.entropy import get_image_entropy_with_highbias
from src.entropy_points import choose_biased_points, get_entropy_points
from src.utils.file_utils import get_image_arr, save_image_to_path
from src.utils.image_utils import (
    draw_points_as_delany_triangles,
    draw_points_on_image,
)


def get_image_save_path(save_dir, filename, blur_radius, num_points, mute_factor, point_candidate_start, step_weight):
    filename = os.path.splitext(os.path.basename(filepath))[0]
    img_name = '{}_{}-blur_{}-pnt_{}-muting_{}-pnt-start_{}-weight.png'.format(filename, blur_radius, num_points, mute_factor, point_candidate_start, step_weight)
    return os.path.join(save_dir, img_name)

def save_points_image(filepath, bw_img, blur_radius, num_points, mute_factor, point_candidate_start, step_weight):
    save_dir = './images/overlayed_points/'
    save_image_to_path(bw_img, get_image_save_path(save_dir, filepath, blur_radius, num_points, mute_factor, point_candidate_start, step_weight))

def save_biased_points_image(filepath, bw_img, blur_radius, num_points, mute_factor, point_candidate_start, step_weight):
    save_dir = './images/biased_overlayed_points/'
    save_image_to_path(bw_img, get_image_save_path(save_dir, filepath, blur_radius, num_points, mute_factor, point_candidate_start, step_weight))

def save_delaney_image(filepath, bw_img, blur_radius, num_points, mute_factor, point_candidate_start, step_weight):
    save_dir = './images/delaney/'
    save_image_to_path(bw_img, get_image_save_path(save_dir, filepath, blur_radius, num_points, mute_factor, point_candidate_start, step_weight))

def save_delaney_entropy_image(filepath, bw_img, blur_radius, num_points, mute_factor, point_candidate_start, step_weight):
    save_dir = './images/delaney/entropy/'
    save_image_to_path(bw_img, get_image_save_path(save_dir, filepath, blur_radius, num_points, mute_factor, point_candidate_start, step_weight))

if __name__ == "__main__":
    n = len(sys.argv)
    print("Total arguments passed: {}".format(n-1))
    if n < 2:
        raise ValueError("Please pass in the filename for the script to use")
    
    env = os.environ.get('ENV', 'DEV-PHOTO')

    filepath = sys.argv[1]
    blur_radius = int(sys.argv[2]) if n > 2 else 5
    num_points = int(sys.argv[3]) if n > 3 else 1000
    mute_factor = float(sys.argv[4]) if n > 4 else 1
    point_candidate_start = int(sys.argv[5]) if n > 5 else 300
    step_weight = float(sys.argv[6]) if n > 6 else 3.0

    # grabbing & loading the image
    bw_img_arr = get_image_arr(filepath, greyscale=True)
    # getting the entropy of the image
    entropy_img_arr = get_image_entropy_with_highbias(bw_img_arr)

    # grab ya some points
    img_points = get_entropy_points(
        np.fliplr(np.rot90(entropy_img_arr, axes=(-1, 0))),
        blur_radius=blur_radius, num_points=num_points * 8,
        mute_factor=mute_factor, show_blur_img=True,
        point_candidate_start=point_candidate_start
    )
    
    biased_points = choose_biased_points(list(img_points), num_selections=num_points, step=step_weight)

    triangle_img_arr = np.zeros(shape=bw_img_arr.shape)
    triangle_img = draw_points_as_delany_triangles(triangle_img_arr, biased_points)
    triangle_entropy_img = draw_points_as_delany_triangles(np.copy(entropy_img_arr), biased_points, draw_color="#808080")
    
    # save and show
    save_delaney_image(filepath, triangle_img, blur_radius, num_points, mute_factor, point_candidate_start, step_weight)
    save_delaney_entropy_image(filepath, triangle_entropy_img, blur_radius, num_points, mute_factor, point_candidate_start, step_weight)

    if env == "DEV-PHOTO":
        entropy_img = Image.fromarray(entropy_img_arr)
        entropy_img.show()

        points_image_arr = draw_points_on_image(np.fliplr(np.rot90(entropy_img_arr, axes=(-1, 0))), img_points, point_radius=2)
        points_image = Image.fromarray(np.fliplr(np.rot90(points_image_arr, axes=(-1, 0))))
        save_points_image(filepath, points_image, blur_radius, num_points, mute_factor, point_candidate_start, step_weight)
        points_image.show()

        biased_points_image_arr = draw_points_on_image(np.fliplr(np.rot90(entropy_img_arr, axes=(-1, 0))), biased_points, point_radius=2)
        biased_points_image = Image.fromarray(np.fliplr(np.rot90(biased_points_image_arr, axes=(-1, 0))))
        save_biased_points_image(filepath, biased_points_image, blur_radius, num_points, mute_factor, point_candidate_start, step_weight)
        biased_points_image.show()

        triangle_img.show()
            
        triangle_entropy_img.show()

    
    """
    # mathplotlib has an inverted y axis
    points_inverted = img_points.T
    img_mask = np.zeros(shape=entropy_img_arr.shape)
    img_mask[points_inverted[0], points_inverted[1]] = 255

    save_image_arr_to_path(
        img_mask, './images/points/white-tiger-dot-child_{}-pnt_{}-muting.png'.format(num_points, mute_factor))


    rotated_points = np.asarray(np.where(np.rot90(img_mask, axes=(1, 0)) != 0)).T
    tri2 = Delaunay(rotated_points)
    delaunay_plot_2d(tri2).show()

    import matplotlib.pyplot as plt
    plt.triplot(rotated_points[:, 0], rotated_points[:, 1], tri2.simplices, marker=None)
    # plt.plot(rotated_points[:, 0], rotated_points[:, 1], '.')
    plt.show()

    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.Delaunay.html
    # https://www.codegrepper.com/code-examples/python/pyplot+save+figure+as+image

    # debugging some stuff
    Image.fromarray(img_mask).show()
    entropy_img_arr[points_inverted[0], points_inverted[1]] = 255
    Image.fromarray(entropy_img_arr).show()
    print(img_points)
    print(tri.points)
    print(tri.simplices)
    """

    print("oof")
