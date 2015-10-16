import cv2
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import eig

def compute_affine_xform(matches, features1, features2, image1, image2):
	"""
	Computer Vision 600.461/661 Assignment 2
	Args:
		matches (list of tuples): list of index pairs of possible matches. For example, if the 4-th feature in feature_coords1 and the 0-th feature
								  in feature_coords2 are determined to be matches, the list should contain (4,0).
        features1 (list of tuples) : list of feature coordinates corresponding to image1
        features2 (list of tuples) : list of feature coordinates corresponding to image2
		image1 (numpy.ndarray): The input image corresponding to features_coords1
		image2 (numpy.ndarray): The input image corresponding to features_coords2
	Returns:
		affine_xform (numpy.ndarray): a 3x3 Affine transformation matrix between the two images, computed using the matches.
	"""

	matches_number = len(matches)

	matched_points_coords_1 = np.zeros((matches_number, 2))
	matched_points_coords_2 = np.zeros((matches_number, 2))

	for ind, match in enumerate(matches):

		img_1_match_ind, img_2_match_ind = match

		matched_points_coords_1[ind, :] = features1[img_1_match_ind]
		matched_points_coords_2[ind, :] = features2[img_2_match_ind]

	points_number = matches_number
	equations_number = 2 * points_number

	inp = matched_points_coords_1
	dst = matched_points_coords_2

	A = np.zeros((equations_number, 7), dtype=np.double)

	points_pairs = zip(inp, dst)

	for pair_number, pair in enumerate(points_pairs):

		x_inp, y_inp = pair[0]
		x_dst, y_dst = pair[1]

		matrix_correspond_row_ind = pair_number * 2

		A[matrix_correspond_row_ind, :] = [x_inp, y_inp, 1, 0, 0, 0, -x_dst]
		A[matrix_correspond_row_ind + 1, :] = [0, 0, 0, x_inp, y_inp, 1, -y_dst]

	solution_matrix = A.T.dot(A)

	eigenvalues, eigenvectors = eig(solution_matrix)

	smallest_eigenvalue_ind = eigenvalues.argsort()[0]
	best_eigenvector = eigenvectors[:, smallest_eigenvalue_ind].copy()

	output_matrix = np.zeros(9)
	output_matrix[8] = best_eigenvector[6]
	best_eigenvector[6] = 0
	best_eigenvector.resize(9)

	output_matrix = output_matrix + best_eigenvector

	output_matrix = output_matrix.reshape((3, 3))

	return output_matrix
