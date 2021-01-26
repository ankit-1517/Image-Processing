import numpy as np
import cv2 as cv

# function to read image at a given path
def readImage(path):
    return cv.imread(path, cv.CV_8UC1)

# function to save image using given image name
def saveImage(img, image_name):
	cv.imwrite(image_name, img)

# function to show image until user pressed any key to close it
def showImage(img):
	cv.imshow('img', img)
	cv.waitKey(0)

# ones-kernel of dimensions size*size
def blockKernel(size):
	return np.ones((size, size), np.uint8)

# disk kernel of radius "size"
def diskKernel(size):
	return cv.getStructuringElement(cv.MORPH_ELLIPSE, (2*size - 1, 2*size - 1))

path = 'line.bmp'
# read image from given path
img = readImage(path)
# apply opening operation to remove lines/rectangles
openImg = cv.morphologyEx(img, cv.MORPH_OPEN, diskKernel(6))
# save output of opening
saveImage(openImg, '2_open.jpg')
# find number of connected components in the image
output = cv.connectedComponentsWithStats(np.uint8(openImg), 4, cv.CV_32S)
num_labels = output[0]
# number of circles = number of connected components - 1
print("Number of circles:",num_labels - 1)
