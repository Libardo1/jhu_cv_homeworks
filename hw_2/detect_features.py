import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d
from scipy.ndimage.filters import gaussian_filter
import skimage.io as io

from nonmaxsuppts import nonmaxsuppts

K = 0.04

def detect_features(image):
	"""
	Computer Vision 600.461/661 Assignment 2
	Args:
		image (numpy.ndarray): The input image to detect features on. Note: this is NOT the image name or image path.
	Returns:
		pixel_coords (list of tuples): A list of (row,col) tuples of detected feature locations in the image
	"""

	harris_threshold = 0.5

	sobel_vertical_kernel = [[-1, 0, 1],
                         [-2, 0, 2],
                         [-1, 0, 1]
                        ]

	sobel_horizontal_kernel = np.rot90(sobel_vertical_kernel)

	I_x = convolve2d(image, sobel_vertical_kernel, mode='same', boundary='symm')
	I_y = convolve2d(image, sobel_horizontal_kernel, mode='same', boundary='symm')

	I_xx = I_x * I_x
	I_yy = I_y * I_y
	I_xy = I_x * I_y

	I_xx = gaussian_filter(I_xx, 3)
	I_yy = gaussian_filter(I_yy, 3)
	I_xy = gaussian_filter(I_xy, 3)

	R = (I_xx * I_yy - I_xy**2) - K*(I_xx + I_yy)**2

	corners = nonmaxsuppts(R, 5, harris_threshold)

	return corners
