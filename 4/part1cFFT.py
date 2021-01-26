# PART 3 DCT
# importing PIL -- useful in reading/writing/viewing images
from PIL import Image 
# importing math -- useful in tan inverse, cos(theta) etc
import math
# importing numpy -- useful in matrix operations
import numpy as np
# importing numpy -- useful in matrix operations
import matplotlib.pyplot as plt
#importing sys library
import sys
# importin g scipy
from scipy import fftpack
# function to read image at a given path
def readImage(path):
    return np.asarray(Image.open(path))

# convert np.array to Image.object
def getImage(img,mode1):
    return Image.fromarray(img, mode = mode1 )

def getArray(img):
    return np.asarray(img)

def saveImage(newimg,mode1,image_name):
	newimg=Image.fromarray(newimg,mode=mode1)	
	newimg.save(image_name)

def MSE(image1 , image2):
	prod = 1
	for i in image1.shape:
		prod *= i 
	temp = image1.astype(np.float64) - image2.astype(np.float64)
	temp = temp*temp
	MSE = np.sum(temp)*1.0/prod
	return MSE

def fourier(im,thereshold = 5):
	print("thereshold: "+ str(thereshold))
	im_fft = fftpack.fft2(im)
	im_fft_mag = np.abs(im_fft)
	count = 0

	count = np.sum(im_fft_mag<thereshold)
	im_fft[im_fft_mag<thereshold] = 0
	decompressed_image = np.abs(fftpack.ifft2(im_fft))
	decompressed_image[decompressed_image > 255] = 255
	decompressed_image[decompressed_image < 0] = 0
	decompressed_image = decompressed_image.astype(np.uint8)
	saveImage(decompressed_image,"RGB","fourier_compression_"+str(thereshold)+"_"+sys.argv[1][:-4]+".jpg")
	
	prod = 1 
	for i in im_fft.shape:
		prod = prod * i

	print("compression_ratio = "+str(((prod)/((prod)-count))))
	saveImage(20*np.log(1+im_fft_mag).astype(np.uint8) , 'RGB' , 'image.jpg')
	print("MSE: "+str(MSE(im,decompressed_image)))

im = readImage(sys.argv[1])
fourier(im, 100)
fourier(im, 200)
fourier(im, 500)
fourier(im, 600)
fourier(im, 700)
fourier(im, 800)
fourier(im, 1000)
fourier(im, 1100)
fourier(im, 1200)
fourier(im, 1300)
fourier(im, 1400)
fourier(im, 1500)
fourier(im, 1500)
fourier(im, 1600)
fourier(im, 1700)
fourier(im, 1800)
fourier(im, 2000)
fourier(im, 2200)
