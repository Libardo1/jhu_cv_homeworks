from scipy.signal import convolve2d
import skimage.io as io
import numpy as np
from scipy.ndimage.filters import gaussian_filter
from math import pi, cos, sin
from matplotlib import pyplot as plt
from skimage.feature import canny
from skimage.draw import line

AMOUNT_OF_BINS_THETA = 50
AMOUNT_OF_BINS_RO = 100


def p5(img):

    img = gaussian_filter(img, 6)

    sobel_vertical_kernel = [[2, 1, 0, -1, -2],
                              [3, 2, 0, -2, -3],
                              [4, 3, 0, -3, -4],
                              [3, 2, 0, -2, -3],
                              [2, 1, 0, -1, -2]]

    sobel_horizontal_kernel = np.rot90(sobel_vertical_kernel)

    vertical_gradient = convolve2d(img, sobel_vertical_kernel, mode='valid')
    horizontal_gradient = convolve2d(img, sobel_horizontal_kernel, mode='valid')

    magnitude = np.sqrt(vertical_gradient**2 + horizontal_gradient**2)

    return magnitude

def p6(edge_image, edge_thresh):

    magnitude = edge_image.copy()

    magnitude[magnitude < edge_thresh] = 0

    img_height, img_width = img.shape

    coords_y, coords_x = magnitude.nonzero()

    theta_ind = np.linspace(-pi/2, pi/2, AMOUNT_OF_BINS_THETA, AMOUNT_OF_BINS_THETA)
    ro_ind, step = np.linspace(0, np.sqrt(img_height**2 + img_width**2), AMOUNT_OF_BINS_RO, retstep=True)

    parameter_matrix = np.zeros((AMOUNT_OF_BINS_THETA, AMOUNT_OF_BINS_RO))

    for coord_num, x in enumerate(coords_x):

        y = coords_y[coord_num]
        ro = y*np.cos(theta_ind) - x*np.sin(theta_ind)

        for ind, value in enumerate(ro):

            if value >= 0:
                parameter_matrix[ind, value // step] += 1

    return magnitude, parameter_matrix

def p7(img, hough_img, hough_threshold):

    hough_img[hough_img < hough_threshold] = 0

    img_height, img_width = thresholded_edge_img.shape

    theta_ind = np.linspace(-pi/2, pi/2, AMOUNT_OF_BINS_THETA)
    ro_ind, step = np.linspace(0, np.sqrt(img_height**2 + img_width**2), AMOUNT_OF_BINS_RO, retstep=True)

    theta_ar, ro_ar = hough_img.nonzero()

    plt.imshow(thresholded_edge_img)
    plt.autoscale(False)

    for line_num in xrange(len(theta_ar)):

        theta = theta_ind[theta_ar[line_num]]
        ro = ro_ind[ro_ar[line_num]]

        tang = sin(theta) / cos(theta)

        b = ro / cos(theta)

        y_pos = img_width*tang + b
        y_neg = b

        try:
            coords_one, coords_two = line(0, int(y_neg), img_width, int(y_pos))
        except:
            continue
        pairs = zip(coords_two, coords_one)

        line_started = False

        for r, c in pairs:

            #print line_started

            smaller_check = r >= 0 and c >= 0
            bigger_check = r < img_height and c < img_width

            if not (smaller_check and bigger_check):
                continue

            if thresholded_edge_img[r, c] and (not line_started):
                line_started = True
                thresholded_edge_img[r, c] = 10000
                plt.plot(c, r, 'ro')
                continue

            if (not thresholded_edge_img[r, c]) and line_started:
                line_started = False
                thresholded_edge_img[r, c] = 10000
                plt.plot(c, r, 'ro')
                continue

        plt.plot([0, img_width], [y_neg, y_pos], color='r', linestyle='-', linewidth=1)

    plt.show()


img = io.imread('hough_complex_1.pgm')


edge_img = p5(img)
thresholded_edge_img, hough_img = p6(edge_img, 500)

p7(img, hough_img, 900)


