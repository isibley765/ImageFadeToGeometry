from PIL import Image, ImageDraw
import numpy as np
from scipy.spatial import Delaunay


def find_square_around_point(point, radius=5):
    # set up the numpy array for the default range
    square_ranges = np.asarray(
        [[-radius, radius+1],
         [-radius, radius+1]]
    )

    # add in the point indexes
    square_ranges[0] += point[0]
    square_ranges[1] += point[1]

    # clean the ranges minimum to 0, negatives act weird
    square_ranges[square_ranges < 0] = 0

    return square_ranges


def draw_points_as_delany_triangles(img_arr, img_points, draw_color="#ffffff"):
    tri = Delaunay(img_points)
    img = Image.fromarray(img_arr)
    draw = ImageDraw.Draw(img)

    for tri_indexes in tri.simplices:
        points = [tuple(tri.points[x]) for x in tri_indexes]
        draw.polygon(points, outline=draw_color)

    return img


def draw_points_on_image(img_arr, img_points, point_radius=1):
    img_arr2 = np.copy(img_arr)

    for point in img_points:
        ranges = find_square_around_point(point, radius=point_radius)
        img_slice = img_arr2[ranges[0][0]:ranges[0][1], ranges[1][0]:ranges[1][1]]
        img_arr2[ranges[0][0]:ranges[0][1], ranges[1][0]:ranges[1][1]] = np.zeros(img_slice.shape, np.uint8)

    return img_arr2
