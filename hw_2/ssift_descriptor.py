import cv2
import numpy
import matplotlib.pyplot as plt

from main import simple_sift, filter_valid_features, fetch_patches

def ssift_descriptor(feature_coords,image):
	"""
	Computer Vision 600.461/661 Assignment 2
	Args:
		feature_coords (list of tuples): list of (row,col) tuple feature coordinates from image
		image (numpy.ndarray): The input image to compute ssift descriptors on. Note: this is NOT the image name or image path.
	Returns:
		descriptors (dictionary{(row,col): 128 dimensional list}): the keys are the feature coordinates (row,col) tuple and
										   the values are the 128 dimensional ssift feature descriptors.
	"""

	patch_size = 40
	grid_number = 4

	features = filter_valid_features(feature_coords, image, patch_size)

	patches = fetch_patches(image, features, patch_size=patch_size)

	patches_1_sift = map(lambda x: simple_sift(x, grid_number), patches)

	output = {}

	for ind, patch_sift in enumerate(patches_1_sift):
		output[feature_coords[ind]] = patch_sift

	return output
