import numpy as np
from scipy.linalg import eig

import skimage.io as io
from skimage.transform import warp

inp = np.asarray([[1, 0], [0, 1], [5, 6], [7, 8], [1, 100], [-1, 500]])
dst = np.asarray([[2, 1], [1, 2], [6, 7], [8, 9], [2, 101], [0, 501]])

def estimate_homography_dlt(inp, dst):

    points_number = inp.shape[0]
    equations_number = 2 * points_number

    A = np.zeros((equations_number, 9), dtype=np.double)

    points_pairs = zip(inp, dst)

    for pair_number, pair in enumerate(points_pairs):

        x_inp, y_inp = pair[0]
        x_dst, y_dst = pair[1]

        matrix_correspond_row_ind = pair_number * 2

        A[matrix_correspond_row_ind, :] = [x_inp, y_inp, 1, 0, 0, 0, -x_dst*x_inp, -x_dst*y_inp, -x_dst]
        A[matrix_correspond_row_ind + 1, :] = [0, 0, 0, x_inp, y_inp, 1, -y_dst*x_inp, -y_dst*y_inp, -y_dst]

    solution_matrix = A.T.dot(A)

    eigenvalues, eigenvectors = eig(solution_matrix)

    smallest_eigenvalue_ind = eigenvalues.argsort()[0]
    best_eigenvector = eigenvectors[:, smallest_eigenvalue_ind]

    output_matrix = best_eigenvector.reshape((3, 3))
    #normailized_matrix = output_matrix / output_matrix[2, 2]

    return output_matrix

def estimate_affine_transformation_dlt(inp, dst):

    points_number = inp.shape[0]
    equations_number = 2 * points_number

    print inp
    print dst

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
    #normailized_matrix = output_matrix / output_matrix[2, 2]

    return output_matrix

def warp_points(points, warp_matrix):

    points_number = points.shape[0]

    homogeneous_coords = np.c_[points, np.ones(points_number)].T

    warped_coords = warp_matrix.dot(homogeneous_coords)

    # Normalized coordinates
    warped_coords = warped_coords / warped_coords[2, :]

    return warped_coords[:2].T

if __name__ == "__main__":

    homography = estimate_homography_dlt(inp, dst)

    res = warp_points(inp, homography)

    print res