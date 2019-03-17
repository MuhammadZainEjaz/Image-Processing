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


def make2D1D(array2d):
    rows, cols = returnRowsAndCols(array2d)
    array1d = np.zeros((rows*cols))
    index=0
    for i in range(0, rows):
        for j in range(0, cols):
            array1d[index] = array2d[i, j]
            index += 1
    return array1d


def applyFilter(window_original_image, filterToApply):
    summation = 0
    rows, cols = returnRowsAndCols(window_original_image)
    for i in range(0, rows):
        for j in range(0, cols):
            summation += (window_original_image[i][j] * filterToApply[i][j])
    return int(summation)


def averageFilter(window_original_image, filterToApply):
    return applyFilter(window_original_image, filterToApply)


def laplacianFilter(window_original_image, filterToApply):
    return applyFilter(window_original_image, filterToApply)


def sobelFilter(window_original_image, filterToApply):
    return applyFilter(window_original_image, filterToApply)


def medianFilter(window_original_image):
    rows, cols = returnRowsAndCols(window_original_image)
    window_original_image = make2D1D(window_original_image)
    window_original_image = np.sort(window_original_image)
    return window_original_image[int(((rows*cols)/2))]


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


def applyMedianFilter(original_image):
    slidingWindowSize = [3, 3]
    padded_image = makeAnewImageWithSomeValues(original_image, slidingWindowSize, 0)
    return traverseWithslidingWindowSize(original_image, slidingWindowSize, padded_image, 0, 1)
    # 1 is the code for median filter


def applyAverageFilter(original_image):
    filterToApply = [[1/9, 1/9, 1/9], [1/9, 1/9, 1/9], [1/9, 1/9, 1/9]]
    slidingWindowSize = [3, 3]
    # filterToApply = [[1/25, 1/25, 1/25, 1/25, 1/25], [1/25, 1/25, 1/25, 1/25, 1/25], [1/25, 1/25, 1/25, 1/25, 1/25],
    #                   [1/25, 1/25, 1/25, 1/25, 1/25], [1/25, 1/25, 1/25, 1/25, 1/25]]
    # slidingWindowSize = [5, 5]
    # filterToApply = [[1/49, 1/49, 1/49, 1/49, 1/49, 1/49, 1/49], [1/49, 1/49, 1/49, 1/49, 1/49, 1/49, 1/49],
    #                  [1/49, 1/49, 1/49, 1/49, 1/49, 1/49, 1/49], [1/49, 1/49, 1/49, 1/49, 1/49, 1/49, 1/49],
    #                  [1/49, 1/49, 1/49, 1/49, 1/49, 1/49, 1/49], [1/49, 1/49, 1/49, 1/49, 1/49, 1/49, 1/49],
    #                  [1/49, 1/49, 1/49, 1/49, 1/49, 1/49, 1/49]]
    # slidingWindowSize = [7, 7]
    padded_image = makeAnewImageWithSomeValues(original_image, slidingWindowSize, 0)
    return traverseWithslidingWindowSize(original_image, slidingWindowSize, padded_image, filterToApply, 2)
    # 2 is the code for averaging filter


def applylaplacian(original_image):
    filterToApply = [[0, 1, 0], [1, -4, 1], [0, 1, 0]]
    slidingWindowSize = [3, 3]
    padded_image = makeAnewImageWithSomeValues(original_image, slidingWindowSize, 0)
    edges = traverseWithslidingWindowSize(original_image, slidingWindowSize, padded_image, filterToApply, 3)
    return original_image - edges
    # 3 is the code for laplacian filter


def applyUnsharpMask(original_image, weight):
    bluredImage = applyAverageFilter(original_image)
    mask = original_image - bluredImage
    return original_image + (weight * mask)


def applySobel(original_image):
    filterToApply = [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]
    slidingWindowSize = [3, 3]
    padded_image = makeAnewImageWithSomeValues(original_image, slidingWindowSize, 0)
    operator1 = traverseWithslidingWindowSize(original_image, slidingWindowSize, padded_image, filterToApply, 4)
    filterToApply = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
    operator2 = traverseWithslidingWindowSize(original_image, slidingWindowSize, padded_image, filterToApply, 4)
    return original_image+(operator1+operator2)


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


image = skio.imread("morphology_set/Fig0905(a)(wirebond-mask).tif", as_gray=True)
image1 = np.asarray(image)
# image = [[2, 3, 4, 75, 67], [26, 35, 44, 53, 26], [52, 13, 41, 54, 61], [22, 13, 41, 35, 46]]
# image1 = np.array(image)

# ------------------------------------------------------------------------------------------------------------------
# This is to test the Morphological part
filter_to_apply = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]
filter_size = [3, 3]
# filter_to_apply = [[1], [1], [1]]
# filter_size = [3, 1]
image1 = negativefunc(image1)
# image2 = apply_erosion(image1, filter_to_apply, filter_size)
# image2 = apply_dillation(image1, filter_to_apply, filter_size)

# image2 = apply_opening(image1, filter_to_apply, filter_size)
# image2 = apply_closing(image1, filter_to_apply, filter_size)

plt.imshow(image1, cmap="gray")
plt.show()
plt.imshow(image2, cmap="gray")
plt.show()
# plt.imshow(image3, cmap="gray")
# plt.show()

# printImage(image1)
# print("\n\n")
# printImage(image2)
