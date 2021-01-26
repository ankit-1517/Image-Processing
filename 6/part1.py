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

# get diamond kernel of dimensions n*n
def diamondKernel(n):
	m = int((n)/2)
	arr = np.zeros((n, n), np.uint8)
	for i in range(n):
		for j in range(n):
			if i+j >= m and abs(i-j) <= m and i + j <= 3*m:
				arr[i][j] = 1
	return arr

# ones-kernel of dimensions size*size
def blockKernel(size):
	return np.ones((size, size), np.uint8)

# disk kernel of radius "size"
def diskKernel(size):
	return cv.getStructuringElement(cv.MORPH_ELLIPSE, (2*size - 1, 2*size - 1))

# take negative of input image
def negative(img, maxVal = 255):
	temp = maxVal*np.ones(img.shape)
	return np.asarray(np.subtract(temp, img), dtype=np.uint8)

path = 'notes.jpg'
# read image from given path
img = readImage(path)
# take negative of input image -- useful in approach 2
neg = negative(img)

# part 1 using given image
# dialation using 5*5 ones block
dilateBlock = cv.dilate(img, blockKernel(5)) 
saveImage(dilateBlock, '1_1_block.jpg')
# dialtion using 5*5 diamond block
dilateDiamond = cv.dilate(img, diamondKernel(5)) 
saveImage(dilateDiamond, '1_1_diamond.jpg')

# part 1 using negative of the image
# dialation using 5*5 ones block
dilateBlock = cv.dilate(neg, blockKernel(5)) 
saveImage(negative(dilateBlock), 'neg_1_1_block.jpg')
# dialtion using 5*5 diamond block
dilateDiamond = cv.dilate(neg, diamondKernel(5)) 
saveImage(negative(dilateDiamond), 'neg_1_1_diamond.jpg')

# part 2 using given image
# closing using 3*3 ones block
closeBlock1 = cv.morphologyEx(img, cv.MORPH_CLOSE, blockKernel(3)) 
saveImage(closeBlock1, '1_2_iter1.jpg')
# closing with 3 iterations using 3*3 ones block
closeBlock2 = cv.morphologyEx(img, cv.MORPH_CLOSE, blockKernel(3))
closeBlock2 = cv.morphologyEx(closeBlock2, cv.MORPH_CLOSE, blockKernel(3))
closeBlock2 = cv.morphologyEx(closeBlock2, cv.MORPH_CLOSE, blockKernel(3))
saveImage(closeBlock2, '1_2_iter3.jpg')

# part 2 using negative of the image
# closing using 3*3 ones block
closeBlock1 = cv.morphologyEx(neg, cv.MORPH_CLOSE, blockKernel(3)) 
saveImage(negative(closeBlock1), 'neg_1_2_iter1.jpg')
# closing with 3 iterations using 3*3 ones block
closeBlock2 = cv.morphologyEx(neg, cv.MORPH_CLOSE, blockKernel(3))
closeBlock2 = cv.morphologyEx(closeBlock2, cv.MORPH_CLOSE, blockKernel(3))
closeBlock2 = cv.morphologyEx(closeBlock2, cv.MORPH_CLOSE, blockKernel(3))
saveImage(negative(closeBlock2), 'neg_1_2_iter3.jpg')

# part 3
# closing usign a disk kernel with radius 2
close = cv.morphologyEx(img, cv.MORPH_CLOSE, diskKernel(2))
saveImage(close, '1_3_close.jpg')
# extract out note heads using thresholding
ret, thresh = cv.threshold(close, 40, 255, cv.THRESH_BINARY)
saveImage(thresh, 'result.jpeg')

