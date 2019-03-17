import sys
import os
import numpy as np
import skimage.io as skio
import matplotlib.pyplot as plt
import skimage as ski

def findMaxAndMin(image):
    min = image[0, 0]
    max = image[0, 0]
    for i in range(0, image.shape[0]):
        for j in range(0, image.shape[1]):
            if image[i, j] < min:
                min = image[i, j]
            if image[i, j] > max:
                max = image[i, j]
    return min, max

def makeLookUpTable(image):
    min, max  = findMaxAndMin(image)
    lookUp = []
    for i in range(0, 256):
        lookUp.append(((i - min)/(max - min)) * 255)
    return lookUp

def contrastStretching(image):
    rows = image.shape[0]
    cols = image.shape[1]
    lookUpTable = makeLookUpTable(image)
    outputImage = np.zeros(image.shape)
    for i in range(0, rows):
        for j in range(0, cols):
            outputImage[i, j] = lookUpTable[image[i, j]]
    return outputImage

def returnRowsAndCols(image):
    return image.shape[0], image.shape[1]


def findPMF(image):
    pmf = []
    for i in range(0, 256):
        pmf.append(0.0)
    rows, cols = returnRowsAndCols(image)
    sum = 0
    for i in range(0, rows):
        for j in range(0, cols):
            pmf[image[i, j]] += 1
            sum += 1
    for i in range(0, 256):
        pmf[i] /= sum
    return pmf

def findCDF(image):
    pmf = findPMF(image)
    cdf = []
    for i in range(0, 256):
        cdf.append(0.0)
    cdf[0] = pmf[0]
    for i in range(1, 256):
        cdf[i] = pmf[i] + cdf[i-1]
    for i in range(0, 256):
        cdf[i] *= 255
    return cdf

def histogramEquilization(image):
    rows, cols = returnRowsAndCols(image)
    cdf = findCDF(image)
    outputImage = np.zeros(image.shape)
    for i in range(0, rows):
        for j in range(0, cols):
            outputImage[i, j] = cdf[image[i, j]]
            # print(outputImage[i, j], " ")
    return outputImage


def countPixels(image):
    count = np.zeros(256)
    rows, cols = returnRowsAndCols(image)
    for i in range(0, rows):
        for j in range(0, cols):
            count[image[i, j]] += 1
    return count


image = skio.imread("Assign_2_ImagePack/pollen.TIF", as_gray=True)
image1 = np.asarray(image)

#image1 = contrastStretching(image1)

# image1 = histogramEquilization(image1)

# rows = image1.shape[0]
# cols = image1.shape[1]
#
# for i in range(0, rows):
#     for j in range(0, cols):
#         print(image1[i, j])

image2 = plt.hist(countPixels(image))

plt.imshow(image2)
plt.show()