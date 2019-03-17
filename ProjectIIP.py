import pydicom as dcm
import sys
import os
import numpy as np
import skimage.io as skio
import matplotlib.pyplot as plt
import skimage as ski
import skimage.morphology as ski_morphology
import skimage.measure as ski_measure
import math
import time

num_of_images = 30
shape_of_image = [0, 0]
# image_name =

def read_images_array():
    # image = np.asarray(dcm.dcmread("dicom/Training/DET0000101/DET0000101_SA1_ph" + str(0) + ".DCM").pixel_array)
    image = np.asarray(dcm.dcmread("dicom/Training/DET0000101/DET0000101_LA1_ph" + str(0) + ".DCM").pixel_array)
    shape_of_image[0] = image.shape[0]
    shape_of_image[1] = image.shape[1]
    image_pack = np.full((num_of_images, image.shape[0], image.shape[1]), 1)
    for i in range(0, num_of_images):
        name = "dicom/Training/DET0000101/DET0000101_LA1_ph" + str(i) + ".DCM"
        # name = "dicom/Training/DET0000101/DET0000101_SA1_ph" + str(i) + ".DCM"
        image = dcm.dcmread(name)
        new = np.asarray(image.pixel_array)
        image_pack[i] = new
    return image_pack


def print_images(image_pack):
    for i in range(0, num_of_images):
        print(image_pack[i], end="\n")


def display_images(image_pack):
    plt.ion()
    for i in range(0, num_of_images):
        plt.imshow(image_pack[i], cmap=plt.cm.Greys_r)
        plt.pause(0.0001)
        plt.show()
        time.sleep(0.30)


def make_a_slice(image_pack, index):
    array = np.full((num_of_images), 1)
    for i in range(0, num_of_images):
        array[i] = image_pack[i, index[0], index[1]]
    return array


def calculate_standard_deviation(image_pack):
    standard_deviation_array = np.full((shape_of_image[0], shape_of_image[1]), 0.0)
    for i in range(0, image_pack[0].shape[0]):
        for j in range(0, image_pack[0].shape[1]):
            standard_deviation_array[i][j] = np.std(make_a_slice(image_pack, (i, j)), ddof=1)
    return standard_deviation_array


def thresh_hold_value(standard_deviation_array, value):
    threshold_value = np.percentile(standard_deviation_array, value)
    return threshold_value


def thresh_holding_func(standard_deviation_array):
    limit = thresh_hold_value(standard_deviation, 93)
    rows, cols = standard_deviation_array.shape[0], standard_deviation_array.shape[1]
    # maximum = np.max(standard_deviation_array)
    # limit = limit / maximum
    new_image = np.zeros(standard_deviation_array.shape)
    # These next steps are just to compute the limit value in floating point
    temp = np.copy(standard_deviation_array)
    temp[0, 0] = limit
    temp = ski.img_as_float(temp)
    limit = temp[0, 0]
    # -----------------------------------------------------------
    original_image = ski.img_as_float(standard_deviation_array)
    for x in range(0, rows):
        for y in range(0, cols):
            if original_image[x, y] >= limit:
                new_image[x, y] = 1
            else:
                new_image[x, y] = 0
    return new_image


# print(image)
image_pack = read_images_array()
# display_images(image_pack)
standard_deviation = calculate_standard_deviation(read_images_array())
thresh_holded_image = thresh_holding_func(standard_deviation)
closed_image = ski_morphology.binary_closing(thresh_holded_image)
label_image = ski_measure.label(closed_image)
regions_image = ski_measure.regionprops(label_image)
find_max_image = np.zeros(len(regions_image))
for i in range(0, len(regions_image)):
    find_max_image[i] = regions_image[i].area
print(np.max(find_max_image))

# plt.imshow(label_image, cmap="gray")
# plt.show()




# print_images(read_images_array())

# sample = np.asarray([[[1, 2], [6, 3], [4, 5]],
#                      [[3, 4], [5, 6], [7, 9]],
#                      [[8, 2], [10, 3], [2, 10]],
#                      [[8, 2], [10, 3], [2, 10]],
#                      [[8, 2], [10, 3], [2, 10]]])
# print(sample[2, 0, 0])


# print(sample[1,0,0])
# print(sample.shape)
# print(np.std(sample[0, :]))
# print("\n")
# print(np.std([1, 6, 4, 3, 5, 7, 8, 10, 2]))
# print(np.std([2, 3, 5]))
# print(np.std([2, 3, 5]))

# for i in range(0, 1):
#     plt.ion()
#     name = "dicom/Training/DET0000101/DET0000101_LA1_ph" + str(i) + ".DCM"
#     image = dcm.dcmread(name)

    # plt.imshow(image.pixel_array, cmap=plt.cm.Greys_r)
    # # plt.figure(image.pixel_array)
    # plt.pause(0.0001)
    # plt.show()
    # time.sleep(0.30)
