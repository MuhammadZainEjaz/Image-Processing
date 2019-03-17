import sys
import os
import numpy as np
import skimage.io as skio
import matplotlib.pyplot as plt
import skimage as ski
import math


def returnRowsAndCols(image):
    return image.shape[0], image.shape[1]


def makeAnewImageWithSomeValues(image, slidingWindowSize, value):
    rows, cols = returnRowsAndCols(image)
    incrementRows = int(slidingWindowSize[0]/2)
    incrementCols = int(slidingWindowSize[1]/2)
    rows += incrementRows*2
    cols += incrementCols*2
    newImage = np.full((rows, cols), value, np.uint8)
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
    return new_image


def applyMedianFilter(original_image):
    slidingWindowSize = [3, 3]
    padded_image = makeAnewImageWithSomeValues(original_image, slidingWindowSize, 0)
    return traverseWithslidingWindowSize(original_image, slidingWindowSize, padded_image, 0, 1)
    # 1 is the code for median filter


def applyAverageFilter(original_image):
    # filterToApply = [[1/9, 1/9, 1/9], [1/9, 1/9, 1/9], [1/9, 1/9, 1/9]]
    # slidingWindowSize = [3, 3]
    filterToApply = [[1/25, 1/25, 1/25, 1/25, 1/25], [1/25, 1/25, 1/25, 1/25, 1/25], [1/25, 1/25, 1/25, 1/25, 1/25],
                      [1/25, 1/25, 1/25, 1/25, 1/25], [1/25, 1/25, 1/25, 1/25, 1/25]]
    slidingWindowSize = [5, 5]
    # filterToApply = [[1/49, 1/49, 1/49, 1/49, 1/49, 1/49, 1/49], [1/49, 1/49, 1/49, 1/49, 1/49, 1/49, 1/49],
    #                  [1/49, 1/49, 1/49, 1/49, 1/49, 1/49, 1/49], [1/49, 1/49, 1/49, 1/49, 1/49, 1/49, 1/49],
    #                  [1/49, 1/49, 1/49, 1/49, 1/49, 1/49, 1/49], [1/49, 1/49, 1/49, 1/49, 1/49, 1/49, 1/49],
    #                  [1/49, 1/49, 1/49, 1/49, 1/49, 1/49, 1/49]]
    # slidingWindowSize = [7, 7]
    padded_image = makeAnewImageWithSomeValues(original_image, slidingWindowSize, 0)
    return traverseWithslidingWindowSize(original_image, slidingWindowSize, padded_image, filterToApply, 2)
    # 2 is the code for averaging filter


def applylaplacian(original_image):
    filterToApply = [[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]]
    slidingWindowSize = [3, 3]
    padded_image = makeAnewImageWithSomeValues(original_image, slidingWindowSize, 0)
    edges = traverseWithslidingWindowSize(original_image, slidingWindowSize, padded_image, filterToApply, 3)
    return np.abs(edges)
    # 3 is the code for laplacian filter


def generateMask(original_image):
    bluredImage = applyAverageFilter(original_image)
    mask = original_image - bluredImage
    return mask


def applyUnsharpMask(original_image, weight):
    mask = generateMask(original_image)
    return original_image + (weight * mask)


def applySobel(original_image):
    filterToApply = [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]
    slidingWindowSize = [3, 3]
    padded_image = makeAnewImageWithSomeValues(original_image, slidingWindowSize, 0)
    operator1 = traverseWithslidingWindowSize(original_image, slidingWindowSize, padded_image, filterToApply, 4)
    filterToApply = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
    operator2 = traverseWithslidingWindowSize(original_image, slidingWindowSize, padded_image, filterToApply, 4)
    return np.abs(operator1*original_image) + np.abs(operator2*original_image)


def powerLaw(originalImage, c, gamma):
    print("min is: ", np.min(originalImage), "\n")
    print("max is: ", np.max(originalImage), "\n")
    originalImage = ski.img_as_float(originalImage)
    rows = originalImage.shape[0]
    cols = originalImage.shape[1]
    newIamge = np.zeros(originalImage.shape)
    for x in range(0, rows-1):
        for y in range(0, cols-1):
            if originalImage[x, y] < 0:
                originalImage[x, y] = 0
            newIamge[x, y] = c * (originalImage[x, y] ** gamma)
    return newIamge


def printImage(image):
    rows, cols = returnRowsAndCols(image)
    for i in range(0, rows):
        for j in range(0, cols):
            print(image[i, j], end=" ")
        print("\n")


image = skio.imread("skeleton.tif", as_gray=True)
image1 = np.asarray(image)

# image = [[2, 3, 4, 75, 67], [26, 35, 44, 53, 26], [52, 13, 41, 54, 61], [22, 13, 41, 35, 46]]
# image1 = np.array(image)

# ------------------------------------------------------------------------------------------------------------------
# Question 1

plt.imshow(image1, cmap="gray")
plt.show()

# ------------------------------------------------------------------------------------------------------------------

# Question 2.a

image2 = applylaplacian(image1)
plt.imshow(image2, cmap="gray")
plt.show()

# ------------------------------------------------------------------------------------------------------------------

# Question 2.b

image3 = image1 + applylaplacian(image1)
plt.imshow(image3, cmap="gray")
plt.show()

# ------------------------------------------------------------------------------------------------------------------

# Question 3.a

image4 = applySobel(image1)
plt.imshow(image4, cmap="gray")
plt.show()

# ------------------------------------------------------------------------------------------------------------------

# Question 3.b

image5 = applyAverageFilter(image4)
plt.imshow(image5, cmap="gray")
plt.show()

# ------------------------------------------------------------------------------------------------------------------

# Question 3.c

image6 = generateMask(image3 * image5)
plt.imshow(image6, cmap="gray")
plt.show()

# ------------------------------------------------------------------------------------------------------------------

# Question 3.d

image7 = image1 + image6
plt.imshow(image7, cmap="gray")
plt.show()

# ------------------------------------------------------------------------------------------------------------------

# Question 3.e

image8 = powerLaw(image7, 1, 0.2)
plt.imshow(image8, cmap="gray")
plt.show()
