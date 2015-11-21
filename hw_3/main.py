import cv, cv2
import numpy as np
import os
import skimage
import skimage.io as io

sift = cv2.SIFT()

def train_classifier(images, number_of_clusters=1000):

    img_number = len(images)

    # Detect and compute Sift descriptor features in each image
    features = map(lambda image: sift.detectAndCompute(image, None)[1], images)

    # Get the index of where for each feature the respective training image label
    # is provided. This is done to compute histogram descriptors of each training sample
    training_set_indexes = np.asarray([])

    for ind, feature in enumerate(features):

        amount_of_elements = feature.shape[0]
        indexes_to_append = np.repeat(ind, amount_of_elements)
        training_set_indexes = np.append(training_set_indexes, indexes_to_append)

    # Put all the descriptors in one array to find clusters (bag of words)
    training_set = np.vstack(features)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

    _, labels, centers = cv2.kmeans(training_set, number_of_clusters, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    # Make it as one dimensional list
    labels = np.concatenate(labels)

    image_descriptors = np.zeros((img_number, number_of_clusters))

    for current_img_number in range(img_number):

        # Get the words of the current image
        current_img_words = labels[training_set_indexes == current_img_number]

        current_img_words_number = len(current_img_words)

        current_histogram = np.bincount(current_img_words, minlength=number_of_clusters).astype(np.float32)

        # Normalize histogram of the current word
        current_histogram = current_histogram / current_img_words_number

        image_descriptors[current_img_number, :] = current_histogram

    return centers, image_descriptors

def get_images_filenames(images_folder, image_categories_folders,
                               amount_of_images_to_take=8, fetch_from_beginnig=True):

    images_filenames_list = []

    for current_category_folder in image_categories_folders:

        # Get the full path to current category folder
        current_category_folder_full_path = os.path.join(images_folder, current_category_folder)

        # Get all the files in current category directory folder
        current_category_filenames = os.listdir(current_category_folder_full_path)

        # Sort all the filename in lexigraphical order. This is to get the filenames
        # sorted like 01.jpg, 02.jpg, 03.jpg and so on.
        current_category_filenames.sort()

        # Take the images from the beginning or from the end.
        if fetch_from_beginnig:
            images_filenames_to_add = current_category_filenames[:amount_of_images_to_take]
        else:
            images_filenames_to_add = current_category_filenames[-amount_of_images_to_take:]

        images_filenames_to_add = map(lambda x: os.path.join(current_category_folder_full_path, x), images_filenames_to_add)

        images_filenames_list.extend(images_filenames_to_add)

    return images_filenames_list



images_folder = 'images'
# image_categories_folders = ['buildings', 'cars', 'faces', 'food', 'people', 'trees']
# amount_of_first_images_to_take = 9

image_categories_folders = ['buildings', 'cars']
amount_of_first_images_to_take = 1

images_filenames = get_images_filenames(images_folder, image_categories_folders,
                                              amount_of_first_images_to_take)

print images_filenames
images = io.imread_collection(images_filenames)

centers, image_descriptors = train_classifier(images)