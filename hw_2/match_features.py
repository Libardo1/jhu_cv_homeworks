import cv2
import numpy
import matplotlib.pyplot as plt

from main import filter_valid_features, fetch_patches, find_best_matches_one_way, find_best_matches_two_way

def match_features(feature_coords1, feature_coords2, image1, image2):
	"""
	Computer Vision 600.461/661 Assignment 2
	Args:
		feature_coords1 (list of tuples): list of (row,col) tuple feature coordinates from image1
		feature_coords2 (list of tuples): list of (row,col) tuple feature coordinates from image2
		image1 (numpy.ndarray): The input image corresponding to features_coords1
		image2 (numpy.ndarray): The input image corresponding to features_coords2
	Returns:
		matches (list of tuples): list of index pairs of possible matches. For example, if the 4-th feature in feature_coords1 and the 0-th feature
								  in feature_coords2 are determined to be matches, the list should contain (4,0).
	"""
	
	patch_size = 30

	features_1 = filter_valid_features(feature_coords1, image1, patch_size)
	features_2 = filter_valid_features(feature_coords2, image2, patch_size)

	patches_1 = fetch_patches(image1, features_1, patch_size=patch_size)
	patches_2 = fetch_patches(image2, features_2, patch_size=patch_size)

	best_matches_1 = find_best_matches_one_way(patches_1, patches_2)
	best_matches_2 = find_best_matches_one_way(patches_2, patches_1)

	matches = find_best_matches_two_way(best_matches_1, best_matches_2)

	return matches
