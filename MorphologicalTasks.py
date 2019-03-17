import sys
import os
import numpy as np
import skimage.io as skio
import matplotlib.pyplot as plt
import skimage as ski
import math


def returnRowsAndCols(image):
    return image.shape[0], image.shape[1]


def negativefunc(originalImage):
    rows = originalImage.shape[0]
    cols = originalImage.shape[1]
    newIamge = np.zeros(originalImage.shape)
    for x in range(0, rows-1):
        for y in range(0, cols-1):
            newIamge[x, y] = (2-1)-originalImage[x, y]
    return newIamge


# Sharpening Part
def makeAnewImageWithSomeValues(image, slidingWindowSize, value):
    rows, cols = returnRowsAndCols(image)
    incrementRows = int(slidingWindowSize[0]/2)
    incrementCols = int(slidingWindowSize[1]/2)
    rows += incrementRows*2
    cols += incrementCols*2
    newImage = np.full((rows, cols), value, image.dtype)
    # newImage = np.full_like(image, value)
    rows, cols = returnRowsAndCols(image)
    for i in range(0, rows):
        for j in range(0, cols):
            newImage[i+incrementRows, j+incrementCols] = image[i, j]
    return newImage


def giveAWindowSizedArrayOnGivenPoint(padded_image, sliding_window_size, index):
    sliding_window_rows, sliding_window_cols = sliding_window_size
    window_fetched = np.zeros(sliding_window_size)
    window_fetched_i, window_fetched_j = 0, 0
    for i in range(index[0], index[0]+sliding_window_rows):
        for j in range(index[1], index[1]+sliding_window_cols):
            window_fetched[window_fetched_i][window_fetched_j] = padded_image[i][j]
            window_fetched_j += 1
        window_fetched_j = 0
        window_fetched_i += 1
    return window_fetched


def applyFilter(window_original_image, filterToApply):
    summation = 0
    rows, cols = returnRowsAndCols(window_original_image)
    for i in range(0, rows):
        for j in range(0, cols):
            summation += (window_original_image[i][j] * filterToApply[i][j])
    return int(summation)


def traverseWithslidingWindowSize(original_image, slidingWindowSize, padded_image, filterToApply, typeOfFilter):
    rows, cols = returnRowsAndCols(original_image)
    new_image = np.zeros(original_image.shape)
    for i in range(0, rows):
        for j in range(0, cols):
            window_fetched = giveAWindowSizedArrayOnGivenPoint(padded_image, slidingWindowSize, (i, j))
            if typeOfFilter == 1:
                new_image[i, j] = medianFilter(window_fetched)
            elif typeOfFilter == 2:
                new_image[i, j] = averageFilter(window_fetched, filterToApply)
            elif typeOfFilter == 3:
                new_image[i, j] = laplacianFilter(window_fetched, filterToApply)
            elif typeOfFilter == 4:
                new_image[i, j] = sobelFilter(window_fetched, filterToApply)
            elif typeOfFilter == 5:
                new_image[i, j] = erosion(window_fetched, filterToApply)
            elif typeOfFilter == 6:
                new_image[i, j] = dillation(window_fetched, filterToApply)
    return new_image

def printImage(image):
    rows, cols = returnRowsAndCols(image)
    for i in range(0, rows):
        for j in range(0, cols):
            print(image[i, j], end=" ")
        print("\n")


# ------------------------------------------------------------------------------------------------------------------
# Morphology Part
def number_of_nonzero_elements(window):
    rows, cols = returnRowsAndCols(np.asarray(window))
    count = 0
    for i in range(0, rows):
        for j in range(0, cols):
            if window[i][j] == 1:
                count += 1
    return count


def fit_hit_miss(window, filter_to_apply):
    summation = applyFilter(window, filter_to_apply)
    n_nz_e = number_of_nonzero_elements(filter_to_apply)
    if summation == n_nz_e:
        return 0    # 0 stands for fit
    elif summation > 0:
        return 1    # 1 stands for hit
    else:
        return 2    # 2 stands for miss


def erosion(window, filter_to_apply):
    action_to_take = fit_hit_miss(window, filter_to_apply)
    if action_to_take == 0:
        return 1
    elif action_to_take == 1:
        return 0
    else:
        return 0


def apply_erosion(original_image, filter_to_apply, filter_size):
    padded_image = makeAnewImageWithSomeValues(original_image, filter_size, 0)
    return traverseWithslidingWindowSize(original_image, filter_size, padded_image, filter_to_apply, 5)
    # 5 stands for erosion


def dillation(window, filter_to_apply):
    action_to_take = fit_hit_miss(window, filter_to_apply)
    if action_to_take == 0:
        return 1
    elif action_to_take == 1:
        return 1
    else:
        return 0


def apply_dillation(original_image, filter_to_apply, filter_size):
    padded_image = makeAnewImageWithSomeValues(original_image, filter_size, 0)
    return traverseWithslidingWindowSize(original_image, filter_size, padded_image, filter_to_apply, 6)
    # 6 stands for dillation


def apply_opening(original_image, filter_to_apply, filter_size):
    return apply_dillation(apply_erosion(original_image, filter_to_apply, filter_size), filter_to_apply, filter_size)


def apply_closing(original_image, filter_to_apply, filter_size):
    return apply_erosion(apply_dillation(original_image, filter_to_apply, filter_size), filter_to_apply, filter_size)


# def apply_region_filling(original_image, point, structuring_element, structuring_element_size):
#     temp =


def make_structuring_element(size, value):
    structuring_element = np.full(size, value)
    return structuring_element


# ------------------------------------------------------------------------------------------------------------------
# Task 1
# 1
image = skio.imread("morphology tasks/fingerprint-1.tif", as_gray=True)
image1 = np.asarray(image)
image1 = ski.img_as_float(image1)
# 2
structuring_element_size = [3, 3]
structuring_element = make_structuring_element(structuring_element_size, 1)
# 3
image2 = apply_opening(image1, structuring_element, structuring_element_size)
# 4
image3 = apply_closing(image2, structuring_element, structuring_element_size)
# 5
# plt.imshow(image1, cmap="gray")
# plt.show()
# plt.imshow(image2, cmap="gray")
# plt.show()
# plt.imshow(image3, cmap="gray")
# plt.show()
printImage(image1)
# ------------------------------------------------------------------------------------------------------------------
# Task 2
# # 1
# image4 = skio.imread("morphology tasks/wires.tif", as_gray=True)
# image5 = np.asarray(image4)
# image5 = ski.img_as_float(image4)
# # 2
# structuring_element_size = [15, 15]
# structuring_element = make_structuring_element(structuring_element_size, 1)
# image6 = apply_closing(image5, structuring_element, structuring_element_size)
# # 3
# structuring_element_size = [53, 53]
# structuring_element = make_structuring_element(structuring_element_size, 1)
# image7 = apply_closing(image5, structuring_element, structuring_element_size)
# # 4
# plt.imshow(image5, cmap="gray")
# plt.show()
# plt.imshow(image6, cmap="gray")
# plt.show()
# plt.imshow(image7, cmap="gray")
# plt.show()
# ------------------------------------------------------------------------------------------------------------------
# Task 3
# # 1
# image8 = skio.imread("morphology tasks/FigP0919(UTK).tif", as_gray=True)
# image9 = np.asarray(image8)
# image9 = ski.img_as_float(image9)
# # 2
# structuring_element_size = [3, 3]
# structuring_element = make_structuring_element(structuring_element_size, 1)
# image10 = apply_erosion(image9, structuring_element, structuring_element_size)
# image11 = image9-image10
# # 3
# plt.imshow(image9, cmap="gray")
# plt.show()
# plt.imshow(image11, cmap="gray")
# plt.show()
# ------------------------------------------------------------------------------------------------------------------
