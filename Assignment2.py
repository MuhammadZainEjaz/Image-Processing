import sys
import os
import numpy as np
import skimage.io as skio
import matplotlib.pyplot as plt
import skimage as ski

def identityfunc(originalImage):
    rows = originalImage.shape[0]
    cols = originalImage.shape[1]
    newIamge = np.zeros(originalImage.shape)
    for x in range(0, rows-1):
        for y in range(0, cols-1):
            newIamge[x, y] = originalImage[x, y]
    return newIamge

def negativefunc(originalImage):
    rows = originalImage.shape[0]
    cols = originalImage.shape[1]
    newIamge = np.zeros(originalImage.shape)
    for x in range(0, rows-1):
        for y in range(0, cols-1):
            newIamge[x, y] = (256-1)-originalImage[x, y]
    return newIamge


def threshholdingfunc(originalImage, limit):
    rows = originalImage.shape[0]
    cols = originalImage.shape[1]
    newIamge = np.zeros(originalImage.shape)
    for x in range(0, rows-1):
        for y in range(0, cols-1):
            if (originalImage[x, y] >= limit):
                newIamge[x, y] = 1
            else:
                newIamge[x, y] = 0
    return newIamge


def imageScallingfunc(originalImage, a):
    rows = originalImage.shape[0]
    cols = originalImage.shape[1]
    newIamge = np.zeros(originalImage.shape)
    for x in range(0, rows-1):
        for y in range(0, cols-1):
            if (a * originalImage[x, y]) >= 255:
                newIamge[x, y] = 255
            else:
                newIamge[x, y] = (a * originalImage[x, y])
    return newIamge

def logTransformationfunc(originalImage, c):
    rows = originalImage.shape[0]
    cols = originalImage.shape[1]
    newIamge = np.zeros(originalImage.shape)
    for x in range(0, rows-1):
        for y in range(0, cols-1):
            newIamge[x, y] = c * np.log((originalImage[x, y] + 1))
    return newIamge

def antiLogTransformationfunc(originalImage, c):
    rows = originalImage.shape[0]
    cols = originalImage.shape[1]
    newIamge = np.zeros(originalImage.shape)
    for x in range(0, rows-1):
        for y in range(0, cols-1):
            newIamge[x, y] = (np.exp(originalImage[x, y]) ** (1/c)) - 1
    return newIamge

def powerLaw(originalImage, c, gamma):
    originalImage = ski.img_as_float(originalImage)
    rows = originalImage.shape[0]
    cols = originalImage.shape[1]
    newIamge = np.zeros(originalImage.shape)
    for x in range(0, rows-1):
        for y in range(0, cols-1):
            newIamge[x, y] = c * (originalImage[x, y] ** gamma)
    return newIamge


def inversePowerLaw(originalImage, c, gamma):
    originalImage = ski.img_as_float(originalImage)
    rows = originalImage.shape[0]
    cols = originalImage.shape[1]
    newIamge = np.zeros(originalImage.shape)
    for x in range(0, rows-1):
        for y in range(0, cols-1):
            newIamge[x, y] = c * (originalImage[x, y] ** (1/gamma))
    return newIamge


def contrastStrechingHelp(originalImage, a1, a2, a3, r1, r2):
    if (originalImage > -1) and (originalImage < r1):
        return originalImage * a1
    elif (originalImage >= r1) and (originalImage <= r2):
        return originalImage * a2
    else:
        if (originalImage * a3) < 256:
            return originalImage * a3
        else:
            return 255

def contrastStreching(originalImage, a1, a2, a3, r1, r2):
    rows = originalImage.shape[0]
    cols = originalImage.shape[1]
    newIamge = np.zeros(originalImage.shape)
    for x in range(0, rows-1):
        for y in range(0, cols-1):
            newIamge[x, y] = contrastStrechingHelp(originalImage[x, y], a1, a2, a3, r1, r2)
    return newIamge

def graylevelslicingHelp(originalImage, a, r1, r2, mode):
    if (originalImage >= r1) and (originalImage <= r2):
        return originalImage * a
    else:
        return originalImage * mode

def grayLevelSlicing(originalImage, a, r1, r2, mode): # Here mode decides whether to off all the other bits or just let them as it is.
    rows = originalImage.shape[0]
    cols = originalImage.shape[1]
    newIamge = np.zeros(originalImage.shape)
    for x in range(0, rows-1):
        for y in range(0, cols-1):
            newIamge[x, y] = graylevelslicingHelp(originalImage[x, y], a, r1, r2, mode)
    return newIamge

image = skio.imread("moon.TIF", as_gray=True)
image1 = np.asarray(image)


#image2 = negativefunc(image1)
#image2 = identityfunc(image1)
#image2 = threshholdingfunc(image1, 200)
#image2 = threshholdingfunc(image1, 10)
#image2 = imageScallingfunc(image1, 1.1)
#image2 = logTransformationfunc(image1, 1)
#image2 = powerLaw(image1, 1, 3.0)
#image2 = inversePowerLaw(image1, 1, 0.6)
#image2 = contrastStreching(image1, 0.7, 1.1, 1.4, 100, 180)
#image2 = grayLevelSlicing(image1,1.5, 100, 150, 1)
# plt.imshow(image2, cmap="gray")
# plt.show()