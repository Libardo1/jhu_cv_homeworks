import numpy as np
import cv2
from matplotlib import pyplot as plt

import numpy as np
import numpy.linalg as linalg
import skimage.io as io

img1 = io.imread('HW3_images/hopkins1.JPG', as_grey=True)
img2 = io.imread('HW3_images/hopkins2.JPG', as_grey=True)

img_1_opencv = cv2.imread('HW3_images/hopkins2.JPG',0)
img_2_opencv = cv2.imread('HW3_images/hopkins2.JPG',0)


def display_matching_points(img_1, img_2, points_img_1, points_img_2):

    matches_number = points_img_1.shape[0]

    concatenated_images = np.concatenate((img_1, img_2), axis=1)
    col_offset = img_1.shape[1]

    plt.imshow(concatenated_images, cmap=plt.get_cmap('gray'))

    for match_number in xrange(matches_number):
        
        img_1_col, img_1_row = points_img_1[match_number, :]
        img_2_col, img_2_row = points_img_2[match_number, :]

        img_2_col = img_2_col + col_offset

        plt.plot([img_1_col, img_2_col], [img_1_row, img_2_row], color='r', linestyle='-', linewidth=1)

    plt.show()

def estimate_fund_matrix(points_left, points_right):
    
    points_number = points_left.shape[0]

    A = np.zeros((points_number, 9), dtype=np.float32)

    for current_point_number in xrange(points_number):

        u_l, v_l = points_left[current_point_number, :]
        u_r, v_r = points_right[current_point_number, :]

        A[current_point_number, :] = [u_l*u_r, u_l*v_r, u_l, v_l*u_r, v_l*v_r, v_l, u_r, v_r, 1]

    matrix_for_estimation = A.T.dot(A)

    eigenValues, eigenVectors = linalg.eig(matrix_for_estimation)

    idx = eigenValues.argsort() 
    eigenValues = eigenValues[idx]
    eigenVectors = eigenVectors[:,idx]


    smallest_eigenvector = eigenVectors[:, 0]

    fund_matrix = smallest_eigenvector.reshape((3, 3))
    
    return fund_matrix
    
    


# Initiate SIFT detector
sift = cv2.SIFT()

# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(img_1_opencv,None)
kp2, des2 = sift.detectAndCompute(img_2_opencv,None)

FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks = 50)

flann = cv2.FlannBasedMatcher(index_params, search_params)

matches = flann.knnMatch(des1,des2,k=2)

# store all the good matches as per Lowe's ratio test.
good = []
for m,n in matches:
    if m.distance < 0.7*n.distance:
        good.append(m)


src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
matchesMask = mask.ravel().tolist()

h,w = img1.shape
pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
dst = cv2.perspectiveTransform(pts,M)


matchesMask = np.asarray(matchesMask)
correct_pair_index = np.where(matchesMask == 1)

src_correct = src_pts[correct_pair_index]
dst_correct = dst_pts[correct_pair_index]

src_correct = np.squeeze(src_correct, axis=1)
dst_correct = np.squeeze(dst_correct, axis=1)

display_matching_points(img1, img2, src_correct, dst_correct)

points_left = src_correct
points_right = dst_correct

fund_matrix = estimate_fund_matrix(points_left, points_right)

print fund_matrix


test_left_point = np.asarray([points_left[0][1], points_left[0][0], 1])

right_image_line_equation = test_left_point.dot(fund_matrix)

#Normalize
right_image_line_equation = right_image_line_equation / right_image_line_equation[2]

right_image_line_equation = np.asarray([right_image_line_equation[1], right_image_line_equation[0], 1])

print right_image_line_equation

first_sample_x = -10
first_eq = right_image_line_equation * first_sample_x
first_eq = first_eq / first_eq[1]
first_output_y = -(first_eq[0] + first_eq[2])

second_sample_x = 1000
second_eq = right_image_line_equation * second_sample_x
second_eq = second_eq / second_eq[1]
second_output_y = -(first_eq[0] + first_eq[2])


# Draw a diagonal blue line with thickness of 5 px
cv2.line(img_2_opencv, (int(first_sample_x), int(first_output_y)), (int(second_sample_x), int(second_output_y)), (255,0,0), 5)

cv2.imshow('image',img_2_opencv)
cv2.waitKey(0)
cv2.destroyAllWindows()