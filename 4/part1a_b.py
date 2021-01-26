# importing PIL -- useful in reading/writing/viewing images
from PIL import Image 
# importing numpy -- useful in matrix operations
import numpy as np
import sys
# print(sys.path)
# sys.path.append('/usr/local/lib/python3.7/site-packages')
# print(sys.path)
import cv2
import matplotlib.pyplot as plt
#import pywt
#import pywt.data

# function to read image at a given path
def readImage(path):
	return np.asarray(Image.open(path))

def saveImg(img, name, colorMap = plt.cm.gray):
	plt.imsave(name + ".jpg", img, cmap=colorMap)

def MSE(image1 , image2):
	prod = 1
	for i in image1.shape:
		prod *= i 
	temp = image1 - image2
	temp = temp*temp
	MSE = np.sum(temp)*1.0/prod
	return MSE


def biLinear(img, name , sx = 2 , sy = 2):
	print(img.shape)
	lSmall = (int(img.shape[1]/sx), int(img.shape[0]/sy))
	lLarge = (img.shape[1], img.shape[0])
	resized = cv2.resize(img, lSmall, interpolation = cv2.INTER_LINEAR)
	print(resized.shape)
	print("compression_ratio_bilinear: "+str(sx*sy))
	saveImg(resized, name + "_bilinear_compress")
	uncompressed = cv2.resize(resized, lLarge, interpolation = cv2.INTER_LINEAR)
	saveImg(uncompressed, name + "_bilinear_uncompress")
	return uncompressed

def biCubic(img, name , sx = 2, sy = 2):
	lSmall = (int(img.shape[1]/sx), int(img.shape[0]/sy))
	lLarge = (img.shape[1], img.shape[0])
	resized = cv2.resize(img, lSmall, interpolation = cv2.INTER_CUBIC)
	saveImg(resized, "_biubic_compress" + name )
	print("compression_ratio_bicubic: "+str(sx*sy))
	uncompressed = cv2.resize(resized, lLarge, interpolation = cv2.INTER_CUBIC)
	saveImg(uncompressed,"_bicubic_uncompress" + name )    
	return uncompressed

image_name = sys.argv[1]
img = readImage(image_name)
image_name = image_name[:-4]
print(image_name)
sx = sys.argv[2]
sy = sys.argv[2]
image_name = image_name + "_" + sx + "_" + sy
sx = float(sx)
sy = float(sy)
x = biLinear(img, image_name , sx , sy)
print(MSE(x , img))
print(x.shape)
x =biCubic(img, image_name , sx , sy)
print(MSE(x , img))
