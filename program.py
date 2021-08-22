import os
import sys

import numpy as np
from PIL import Image
from scipy.spatial import Delaunay, delaunay_plot_2d

from src.entropy import get_image_entropy_with_highbias
from src.entropy_points import draw_points_as_delany_triangles, get_entropy_points
from src.utils.file_utils import get_image_arr, save_image_arr_to_path, save_image_to_path

def save_entropy_image(filepath, bw_img, num_points, mute_factor):
    save_dir = './images/triangles'
    filename = os.path.splitext(os.path.basename(filepath))[0]
    save_path = os.path.join(
        save_dir, '{}_{}-pnt_{}-muting.png'.format(filename, num_points, mute_factor))
    save_image_to_path(bw_img, save_path)

if __name__ == "__main__":
    n = len(sys.argv)
    print("Total arguments passed: {}".format(n-1))
    if n < 2:
        raise ValueError("Please pass in the filename for the script to use")

    filepath = sys.argv[1]
    blur_radius = int(sys.argv[2]) if n > 2 else 5
    num_points = int(sys.argv[3]) if n > 3 else 1000
    mute_factor = float(sys.argv[4]) if n > 4 else 1

    # grabbing & loading the image
    bw_img_arr = get_image_arr(filepath, greyscale=True)
    # getting the entropy of the image
    entropy_img_arr = get_image_entropy_with_highbias(bw_img_arr)

    # grab ya some points
    img_points = get_entropy_points(
        np.fliplr(np.rot90(entropy_img_arr, axes=(-1, 0))),
        blur_radius=blur_radius,
        num_points=num_points, mute_factor=mute_factor
    )
    bw_img_arr2 = np.zeros(shape=bw_img_arr.shape)
    bw_img = draw_points_as_delany_triangles(bw_img_arr2, img_points)
    
    # save and show
    save_entropy_image(filepath, bw_img, num_points, mute_factor)
    bw_img.show()
    
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
