import cv, cv2
import numpy as np
import os
import skimage
import skimage.io as io
from skimage.transform import rescale
from skimage.color import rgb2gray
from skimage import img_as_ubyte

sift = cv2.SIFT()

biggest_axis_threshold = 512

def image_rescale_loader(image_path):
    
    image = io.imread(image_path)
    
    rows_number = image.shape[0]
    cols_number = image.shape[1]

    biggest_axis = rows_number if rows_number > cols_number else cols_number

    biggest_axis = rows_number if rows_number > cols_number else cols_number

    if biggest_axis > biggest_axis_threshold:

        ratio = biggest_axis_threshold / float(biggest_axis)

        image = rescale(image, ratio)
    
    return img_as_ubyte(rgb2gray(image))

def compute_images_sift_features(images):
    
    return map(lambda image: sift.detectAndCompute(image, None)[1], images)

def train_classifier(images, number_of_clusters=1000):

    img_number = len(images)

    # Detect and compute Sift descriptor features in each image
    features = compute_images_sift_features(images)

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

    # Make labels array as one dimensional list
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
    
    number_of_categories = len(image_categories_folders)
    
    labels = np.repeat(np.arange(number_of_categories), amount_of_images_to_take)
    
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

    return images_filenames_list, labels

def closest_vector(vector_to_check, vectors_to_compare_to):
    
    return np.argmin(((vector_to_check - vectors_to_compare_to)**2).sum(axis=1))

def closest_vector_batch(vectors_to_check, vectors_to_compare_to):
    
    return np.asarray(map(lambda x: closest_vector(x, vectors_to_compare_to), vectors_to_check))

def compute_bag_of_words_repr(feature_vect, clusters):
    
    number_of_clusters = clusters.shape[0]
    
    img_words_number = feature_vect.shape[0]
    
    bag_of_words_count = closest_vector_batch(feature_vect, clusters)
    
    bag_of_words_count = np.bincount(bag_of_words_count, minlength=number_of_clusters).astype(np.float32)
    
    bag_of_words_frequency = bag_of_words_count / img_words_number
    
    return bag_of_words_frequency

def compute_bag_of_words_repr_batch(feature_vectors, clusters):
    
    result = map(lambda x: compute_bag_of_words_repr(x, clusters), feature_vectors)
    
    return np.vstack(result)

def images_to_bag_of_words_histogram(images, clusters):
    
    images_features = compute_images_sift_features(images)
    images_histograms = compute_bag_of_words_repr_batch(images_features, clusters)
    
    return images_histograms

def nearest_neighbour_classifier(train_features, train_features_labels, test_features):
    
    res = closest_vector_batch(test_features, train_features)
    
    return train_features_labels[res]

def get_classification_accuracy(ground_truth_labels, check_labels):
    
    return (ground_truth_labels == check_labels).sum() / float(check_labels.shape[0])

images_folder = 'images'
image_categories_folders = ['buildings', 'cars', 'faces', 'food', 'people', 'trees']
number_of_images = 11
number_of_train_images = 9
number_of_test_images = number_of_images - number_of_train_images

train_images_filenames, train_images_class_labels = get_images_filenames(images_folder, image_categories_folders,
                                                                         number_of_train_images)

test_images_filenames, test_images_class_labels = get_images_filenames(images_folder, image_categories_folders,
                                                                        number_of_test_images, fetch_from_beginnig=False)

print train_images_filenames
print test_images_filenames

train_images = io.ImageCollection(train_images_filenames, load_func=image_rescale_loader)
test_images = io.ImageCollection(test_images_filenames, load_func=image_rescale_loader)

cluster_centers, train_images_histograms = train_classifier(train_images, number_of_clusters=100)

test_images_histograms = images_to_bag_of_words_histogram(test_images, cluster_centers)

clas = nearest_neighbour_classifier(train_images_histograms, train_images_class_labels, test_images_histograms)
real = test_images_class_labels

print get_classification_accuracy(real, clas)