import numpy as np
import skimage.io as io
from math import atan2, cos, sin, pi

from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec

from lib import UnionFind

def p1(gray_img, threshold):

    output = np.zeros_like(gray_img)
    above_threshold_ind = (gray_img > threshold)
    output[above_threshold_ind] = 1

    return output

def get_neighbours_labels(component_img, row, col):

    # Default values: background values
    neighbours_labels = [0, 0, 0]

    height, width = component_img.shape

    # Neighbours: Left, Top-Left, Top
    neighbours = ((0, -1), (-1, -1), (-1, 0))

    for (neighbour_num, (height_shift, width_shift)) in enumerate(neighbours):

        neighbour_row = row + height_shift
        neighbour_col = col + width_shift

        passes_bounds_check = (neighbour_row < height) and (neighbour_col < width)
        passes_bounds_check &= (neighbour_row >= 0) and (neighbour_col >= 0)

        # Count out-of-bounds elements as background elements
        if not passes_bounds_check:
            continue

        current_label = component_img[neighbour_row, neighbour_col]

        neighbours_labels[neighbour_num] = current_label

    return neighbours_labels


def p2(binary_img):

    union_find = UnionFind()

    # Negate to make the the interest pixels equal to -1
    component_img = binary_img * -1
    height, width = binary_img.shape
    current_label = 0

    for row in xrange(height):
        for col in xrange(width):

            if component_img[row, col] == -1:

                neighbours_labels = get_neighbours_labels(component_img, row, col)

                # First case from slides:
                # All neighbours are background or out of bounds.
                # Create new label for the element.
                if neighbours_labels == [0, 0, 0]:

                    current_label += 1
                    component_img[row, col] = current_label
                    continue

                # Second case from slides:
                # Top left element has class label.
                # Assign current element to it.
                if neighbours_labels[1] != 0:

                    component_img[row, col] = neighbours_labels[1]
                    continue

                # Third case from slides:
                # Both top elements are zero.
                # Assign left element's value to the element.
                if neighbours_labels[2] == 0:

                    component_img[row, col] = neighbours_labels[0]
                    continue

                # Forth case from slides:
                # Both left elements are zero.
                # Assign top element's value to the element.
                if neighbours_labels[0] == 0:

                    component_img[row, col] = neighbours_labels[2]
                    continue

                # Fifth case from slides:
                # Left and top elements are not zero and equal.
                # Assign element to this value.
                if neighbours_labels[0] == neighbours_labels[2]:

                    component_img[row, col] = neighbours_labels[0]
                    continue

                component_img[row, col] = neighbours_labels[2]

                union_find.union(neighbours_labels[0], neighbours_labels[2])

    # Second pass to resolve the equivalent elements.
    current_max_label = 0
    saved_labels = {}

    for row in xrange(height):
        for col in xrange(width):

            current_label = component_img[row, col]

            if current_label != 0:

                current_label_class = union_find[current_label]

                if current_label_class not in saved_labels:

                    current_max_label += 1
                    saved_labels[current_label_class] = current_max_label

                component_img[row, col] = saved_labels[current_label_class]

    # Create overlays for each component
    output_layers = []
    labels = np.unique(component_img)
    label_numbers = labels[1:]

    for number in label_numbers:

        output_layers.append(component_img == number)

    return output_layers

def p3(overlays):

    database = []

    for elements in overlays:

        object_descriptor = {}
        database.append(object_descriptor)

        A = elements.sum()

        coords_i, coords_j = elements.nonzero()

        coords_j_sum = coords_j.sum()
        coords_i_sum = coords_i.sum()

        x_bar = coords_j_sum / float(A)
        y_bar = coords_i_sum / float(A)

        a_prime = (coords_i**2).sum()
        b_prime = 2*(coords_i*coords_j).sum()
        c_prime = (coords_j**2).sum()

        a = a_prime - (y_bar**2)*A
        b = b_prime - 2*x_bar*y_bar*A
        c = c_prime - (x_bar**2)*A

        theta = atan2(b, a - c) / 2

        E_min = a*sin(theta)**2 - b*sin(theta)*cos(theta) + c*cos(theta)**2

        new_angle = theta + (pi/2)

        E_max = a*sin(new_angle)**2 - b*sin(new_angle)*cos(new_angle) + c*cos(new_angle)**2

        roundness = E_min / E_max

        object_descriptor['x_position'] = x_bar
        object_descriptor['y_position'] = y_bar
        object_descriptor['min_moment'] = E_min
        object_descriptor['orientation'] = theta * (180 / pi)
        object_descriptor['roundness'] = roundness

    return database

def p4():

    img_1 = io.imread('two_objects.pgm', as_grey=True)
    img_2 = io.imread('many_objects_2.pgm', as_grey=True)

    labels_1 = p2(p1(img_1, 110))
    labels_2 = p2(p1(img_2, 110))

    database_1 = p3(labels_1)
    database_2 = p3(labels_2)

    matches = []

    for current_obj_num, current_obj in enumerate(database_1):

        current_roundness = current_obj['roundness']

        for compared_obj_num, compared_obj in enumerate(database_2):

            compared_roundness = compared_obj['roundness']

            if abs(current_roundness - compared_roundness) < 0.04:

                matches.append((current_obj_num, compared_obj_num))

    matches_amount = len(matches)

    fig = plt.figure()
    # create figure window

    gs = gridspec.GridSpec(matches_amount, 2)
    # Creates grid 'gs' of a rows and b columns

    for (match_num, (original_obj_num, matched_obj_num)) in enumerate(matches):

        original_x = database_1[original_obj_num]['x_position']
        original_y = database_1[original_obj_num]['y_position']
        original_orient = database_1[original_obj_num]['orientation'] * (pi/180)

        matched_x = database_2[matched_obj_num]['x_position']
        matched_y = database_2[matched_obj_num]['y_position']
        matched_orient = database_2[matched_obj_num]['orientation'] * (pi/180)

        ax = plt.subplot(gs[match_num-1, 0])
        # Adds subplot 'ax' in grid 'gs' at position [x,y]
        ax.imshow(img_1)
        ax.plot(original_x, original_y, 'ro')

        step = 30
        tang = cos(original_orient) / sin(original_orient)

        y_pos = step*tang
        y_neg = -step*tang

        ax.plot([original_x - step, original_x + step], [original_y + y_neg, original_y + y_pos], color='b', linestyle='-', linewidth=2)

        ax2 = plt.subplot(gs[match_num-1, 1])
        # Adds subplot 'ax' in grid 'gs' at position [x,y]
        ax2.imshow(img_2)
        ax2.plot(matched_x, matched_y, 'ro')

        step = 30
        tang = cos(matched_orient) / sin(matched_orient)

        y_pos = step*tang
        y_neg = -step*tang

        ax2.plot([matched_x - step, matched_x + step], [matched_y + y_neg, matched_y + y_pos], color='b', linestyle='-', linewidth=2)

        fig.add_subplot(ax)

    plt.show()

if __name__ == "__main__":

    p4()



