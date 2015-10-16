import numpy as np
from scipy.signal import convolve2d
from scipy.ndimage.filters import gaussian_filter
import skimage.io as io
from skimage.transform import warp, ProjectiveTransform, SimilarityTransform

from skimage.color import rgb2gray

from matplotlib import pyplot as plt

from nonmaxsuppts import nonmaxsuppts

from lib import estimate_homography_dlt, estimate_affine_transformation_dlt
from numpy.random import choice

from operator import add

from numpy.linalg import norm

K = 0.04
ssift_threshold = 0.75

sobel_vertical_kernel = [[-1, 0, 1],
                         [-2, 0, 2],
                         [-1, 0, 1]
                        ]

sobel_horizontal_kernel = np.rot90(sobel_vertical_kernel)


def detect_features(img, harris_threshold=0.5):

    I_x = convolve2d(img, sobel_vertical_kernel, mode='same', boundary='symm')
    I_y = convolve2d(img, sobel_horizontal_kernel, mode='same', boundary='symm')

    I_xx = I_x * I_x
    I_yy = I_y * I_y
    I_xy = I_x * I_y

    I_xx = gaussian_filter(I_xx, 3)
    I_yy = gaussian_filter(I_yy, 3)
    I_xy = gaussian_filter(I_xy, 3)

    R = (I_xx * I_yy - I_xy**2) - K*(I_xx + I_yy)**2

    corners = nonmaxsuppts(R, 5, harris_threshold)

    return corners

def is_feature_valid(row, col, img, patch_size):

    img_row_size, img_col_size = img.shape

    begin_row_ind = row - patch_size
    end_row_ind = row + patch_size + 1

    begin_col_ind = col - patch_size
    end_col_ind = col + patch_size + 1

    row_bounds_check = (begin_row_ind >= 0) & (end_row_ind < img_row_size)
    col_bounds_check = (begin_col_ind >= 0) & (end_col_ind < img_col_size)

    return row_bounds_check & col_bounds_check


def filter_valid_features(features, img, patch_size):

    return filter(lambda coords: is_feature_valid(coords[0], coords[1], img, patch_size), features)

def fetch_patches(img, features, patch_size):

    patches = []

    for feature in features:

        row = feature[0]
        col = feature[1]

        begin_row_ind = row - patch_size
        end_row_ind = row + patch_size + 1

        begin_col_ind = col - patch_size
        end_col_ind = col + patch_size + 1

        patch = img[begin_row_ind:end_row_ind, begin_col_ind:end_col_ind]
        patches.append(patch)

    return patches

def squared_sum_distance(patch_1, patch_2):

    return ((patch_1 - patch_2)**2).sum()

def find_best_matches_one_way(patches_1, patches_2):

    patches_1_best_pairs = np.zeros(len(patches_1), dtype=np.int)

    for ind, patch in enumerate(patches_1):

        comarison_results = np.asarray(map(lambda x: squared_sum_distance(patch, x), patches_2))

        patches_1_best_pairs[ind] = np.argmin(comarison_results)

    return patches_1_best_pairs

def find_best_matches_two_way(best_matches_1, best_matches_2):

    matched_points_ind = []

    for patch_1_ind, patch_2_ind in enumerate(best_matches_1):

        # If the features mutually best for each other
        if best_matches_2[patch_2_ind] == patch_1_ind:
            matched_points_ind.append((patch_1_ind, patch_2_ind))

    return matched_points_ind

def detect_common_features_patch(img_1, img_2, patch_size=30, harris_threshold=0.5):

    features_1 = filter_valid_features(detect_features(img_1, harris_threshold), img_1, patch_size)
    features_2 = filter_valid_features(detect_features(img_2, harris_threshold), img_2, patch_size)

    patches_1 = fetch_patches(img_1, features_1, patch_size=patch_size)
    patches_2 = fetch_patches(img_2, features_2, patch_size=patch_size)

    best_matches_1 = find_best_matches_one_way(patches_1, patches_2)
    best_matches_2 = find_best_matches_one_way(patches_2, patches_1)

    matches = find_best_matches_two_way(best_matches_1, best_matches_2)

    matches_number = len(matches)

    matched_points_coords_1 = np.zeros((matches_number, 2))
    matched_points_coords_2 = np.zeros((matches_number, 2))

    for ind, match in enumerate(matches):

        img_1_match_ind, img_2_match_ind = match

        matched_points_coords_1[ind, :] = features_1[img_1_match_ind]
        matched_points_coords_2[ind, :] = features_2[img_2_match_ind]

    return matched_points_coords_1, matched_points_coords_2

def display_matching_points(img_1, img_2, points_img_1, points_img_2):

    matches = zip(points_img_1, points_img_2)

    concatenated_images = np.concatenate((img_1, img_2), axis=1)
    col_offset = img_1.shape[1]

    plt.imshow(concatenated_images, cmap=plt.get_cmap('gray'))

    for point_1, point_2 in matches:

        img_1_row, img_1_col = point_1
        img_2_row, img_2_col = point_2

        img_2_col = img_2_col + col_offset

        plt.plot([img_1_col, img_2_col], [img_1_row, img_2_row], color='r', linestyle='-', linewidth=1)

    plt.show()

def warp_points(points, warp_matrix):

    points_number = points.shape[0]

    homogeneous_coords = np.c_[points, np.ones(points_number)].T

    warped_coords = warp_matrix.dot(homogeneous_coords)

    # Normalized coordinates
    warped_coords = warped_coords / warped_coords[2, :]

    return warped_coords[:2].T

def ransac_dlt(points_img_1, points_img_2, sample_size=4, iteration_amount=1000, threshold=50):

    matched_points_number = points_img_1.shape[0]

    indexes = range(matched_points_number)

    best_consensus_count = 0

    # Should create a list with False instead. Can cause error later.
    best_consensus_index = []

    for i in range(iteration_amount):

        random_index = choice(indexes, size=sample_size, replace=False)

        random_points_inp = points_img_1[random_index]
        random_points_dst = points_img_2[random_index]

        warp_matrix = estimate_affine_transformation_dlt(random_points_inp, random_points_dst)

        dst_test_warp = warp_points(points_img_1, warp_matrix)

        points_distances = points_img_2 - dst_test_warp

        points_distances = np.sqrt((points_distances**2).sum(axis=1))

        consensus_index = points_distances < threshold

        consensus_number = np.count_nonzero(consensus_index)

        if consensus_number > best_consensus_count:
            best_consensus_count = consensus_number
            best_consensus_index = consensus_index

    points_img_1_selected = points_img_1[best_consensus_index]
    points_img_2_selected = points_img_2[best_consensus_index]

    estimated_homography = estimate_affine_transformation_dlt(points_img_1_selected, points_img_2_selected)

    return points_img_1_selected, points_img_2_selected, estimated_homography


def create_panorama_patch(img_1_orig, img_2_orig):

    img_1 = rgb2gray(img_1_orig)
    img_2 = rgb2gray(img_2_orig)

    points_img_1, points_img_2 = detect_common_features_patch(img_1, img_2, patch_size=30, harris_threshold=1)

    #display_matching_points(img_1_orig, img_2_orig, points_img_1, points_img_2)

    points_img_1_xy = np.fliplr(points_img_1)
    points_img_2_xy = np.fliplr(points_img_2)

    points_img_1_xy_selected, points_img_2_xy_selected, warp_matrix = ransac_dlt(points_img_1_xy, points_img_2_xy, iteration_amount=1000)

    points_img_1_selected = np.fliplr(points_img_1_xy_selected)
    points_img_2_selected = np.fliplr(points_img_2_xy_selected)

    #display_matching_points(img_1_orig, img_2_orig, points_img_1_selected, points_img_2_selected)

    transform_object = ProjectiveTransform(warp_matrix)

    # Find the borders after the first image transformation to scale window.
    img_1_row_size, img_1_col_size = img_1.shape[:2]

    img_1_corner_coords = np.asarray([[0, 0],
                           [0, img_1_col_size],
                           [img_1_row_size, img_1_col_size],
                           [img_1_row_size, 0]])

    img_1_corner_coords_xy = np.fliplr(img_1_corner_coords)

    # Find the borders after the second image transformation to scale window.
    img_2_row_size, img_2_col_size = img_2.shape[:2]

    img_2_corner_coords = np.asarray([[0, 0],
                           [0, img_2_col_size],
                           [img_2_row_size, img_2_col_size],
                           [img_2_row_size, 0]])

    img_2_corner_coords_xy = np.fliplr(img_2_corner_coords)

    img_1_corner_coords_warped_xy = transform_object(img_1_corner_coords_xy)

    img_1_corner_coords_warped_xy = img_1_corner_coords_warped_xy.astype(np.int)

    all_corners = np.vstack((img_1_corner_coords_warped_xy, img_2_corner_coords_xy))

    corner_min = np.min(all_corners, axis=0)
    corner_max = np.max(all_corners, axis=0)

    output_shape = (corner_max - corner_min)
    output_shape += np.abs(corner_min)

    offset = SimilarityTransform(translation=-corner_min)

    print corner_min

    output_shape = output_shape[::-1]

    img_1_final = warp(img_1_orig, (transform_object + offset).inverse, output_shape=output_shape)
    img_2_final = warp(img_2_orig, offset.inverse, output_shape=output_shape)

    print img_1_final.shape, img_2.shape

    img_final = (img_1_final / 2) + (img_2_final / 2)

    io.imshow(img_final)
    io.show()


def split_matrix_into_blocks(matrix, splits_number):

    # Split along first dimension
    half_split = np.array_split(matrix, splits_number)

    # Split along second direction
    res = map(lambda x: np.array_split(x, splits_number, axis=1), half_split)

    # Concatenate nested arrays
    res = reduce(add, res)

    return res

def simple_sift(patch, grid_number):

    I_x = convolve2d(patch, sobel_vertical_kernel, mode='same', boundary='symm')
    I_y = convolve2d(patch, sobel_horizontal_kernel, mode='same', boundary='symm')

    sift_orientation_histogram_bins = np.linspace(-np.pi, np.pi, 9)

    magnitude_weights = np.sqrt(I_x**2 + I_y**2)

    orientation = np.arctan2(I_y, I_x)

    magnitude_weights_blocks = split_matrix_into_blocks(magnitude_weights, grid_number)
    orientation_blocks = split_matrix_into_blocks(orientation, grid_number)

    descriptor = []

    for magnitude_weight_block, orientation_block in zip(magnitude_weights_blocks, orientation_blocks):

        block_descriptor, _ = np.histogram(orientation_block, bins=sift_orientation_histogram_bins, weights=magnitude_weight_block)

        descriptor.extend(block_descriptor)

    descriptor = np.asarray(descriptor)
    descriptor = descriptor / norm(descriptor)
    descriptor[descriptor > 0.2] = 0.2
    descriptor = descriptor / norm(descriptor)

    return descriptor

def detect_common_features_ssift(img_1, img_2, patch_size=40, grid_number=4, harris_threshold=0.5):

    features_1 = filter_valid_features(detect_features(img_1, harris_threshold), img_1, patch_size)
    features_2 = filter_valid_features(detect_features(img_2, harris_threshold), img_2, patch_size)

    patches_1 = fetch_patches(img_1, features_1, patch_size=patch_size)
    patches_2 = fetch_patches(img_2, features_2, patch_size=patch_size)

    patches_1_sift = map(lambda x: simple_sift(x, grid_number), patches_1)
    patches_2_sift = map(lambda x: simple_sift(x, grid_number), patches_2)

    matches = []

    for ind, patch in enumerate(patches_1_sift):

        comparison_results = np.asarray(map(lambda x: squared_sum_distance(patch, x), patches_2_sift))

        sorted_index = np.argsort(comparison_results)

        ratio_test = comparison_results[sorted_index[0]] / float(comparison_results[sorted_index[1]])

        if ratio_test <= ssift_threshold:
            matches.append((ind, sorted_index[0]))

    matches_number = len(matches)

    matched_points_coords_1 = np.zeros((matches_number, 2))
    matched_points_coords_2 = np.zeros((matches_number, 2))

    for ind, match in enumerate(matches):

        img_1_match_ind, img_2_match_ind = match

        matched_points_coords_1[ind, :] = features_1[img_1_match_ind]
        matched_points_coords_2[ind, :] = features_2[img_2_match_ind]

    return matched_points_coords_1, matched_points_coords_2

def create_panorama_ssift(img_1_orig, img_2_orig):

    img_1 = rgb2gray(img_1_orig)
    img_2 = rgb2gray(img_2_orig)

    points_img_1, points_img_2 = detect_common_features_ssift(img_1, img_2, harris_threshold=0.05)

    #display_matching_points(img_1_orig, img_2_orig, points_img_1, points_img_2)

    points_img_1_xy = np.fliplr(points_img_1)
    points_img_2_xy = np.fliplr(points_img_2)

    points_img_1_xy_selected, points_img_2_xy_selected, warp_matrix = ransac_dlt(points_img_1_xy, points_img_2_xy, iteration_amount=1000)

    points_img_1_selected = np.fliplr(points_img_1_xy_selected)
    points_img_2_selected = np.fliplr(points_img_2_xy_selected)

    #display_matching_points(img_1_orig, img_2_orig, points_img_1_selected, points_img_2_selected)

    transform_object = ProjectiveTransform(warp_matrix)

    # Find the borders after the first image transformation to scale window.
    img_1_row_size, img_1_col_size = img_1.shape[:2]

    img_1_corner_coords = np.asarray([[0, 0],
                           [0, img_1_col_size],
                           [img_1_row_size, img_1_col_size],
                           [img_1_row_size, 0]])

    img_1_corner_coords_xy = np.fliplr(img_1_corner_coords)

    # Find the borders after the second image transformation to scale window.
    img_2_row_size, img_2_col_size = img_2.shape[:2]

    img_2_corner_coords = np.asarray([[0, 0],
                           [0, img_2_col_size],
                           [img_2_row_size, img_2_col_size],
                           [img_2_row_size, 0]])

    img_2_corner_coords_xy = np.fliplr(img_2_corner_coords)

    img_1_corner_coords_warped_xy = transform_object(img_1_corner_coords_xy)

    img_1_corner_coords_warped_xy = img_1_corner_coords_warped_xy.astype(np.int)

    all_corners = np.vstack((img_1_corner_coords_warped_xy, img_2_corner_coords_xy))

    corner_min = np.min(all_corners, axis=0)
    corner_max = np.max(all_corners, axis=0)

    output_shape = (corner_max - corner_min)
    output_shape += np.abs(corner_min)

    offset = SimilarityTransform(translation=-corner_min)

    print corner_min

    output_shape = output_shape[::-1]

    img_1_final = warp(img_1_orig, (transform_object + offset).inverse, output_shape=output_shape)
    img_2_final = warp(img_2_orig, offset.inverse, output_shape=output_shape)

    print img_1_final.shape, img_2.shape

    img_final = (img_1_final / 2) + (img_2_final / 2)

    io.imshow(img_final)
    io.show()

if __name__ == '__main__':

    img_1_orig = io.imread('bikes1.png')
    img_2_orig = io.imread('bikes2.png')

    img_1 = rgb2gray(img_1_orig)
    img_2 = rgb2gray(img_2_orig)

    create_panorama_patch(img_1, img_2)
