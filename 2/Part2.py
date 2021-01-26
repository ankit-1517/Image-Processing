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

# function to construct histogram
def construct_histogram(img):
	# intensity ranges from [0, 255]
	histogram = [0]*256
	for j in range(img.shape[1]):
		for i in range(img.shape[0]):
			# count no of pixels with intensity img[i][j]
			histogram[int(img[i][j])] += 1
	return histogram

# function to find the cummulative distribution function of image 
def find_CDF(histogram, total_pixels):
	histogram = [i*1.0/total_pixels for i in histogram]
	CDF = [0]*256
	temp = 0
	for i in range(256):
		temp += histogram[i]
		CDF[i] = temp
	return CDF

# function to plot bar graph with x, y values
def plot_bar(y_values, xlabel, ylabel, title, name):
	B = range(256)
	plt.bar(B,y_values, color='g')
	plt.xlabel(xlabel)
	plt.ylabel(ylabel)
	plt.title(title)
	plt.savefig(name) 
	
# mapping used in contrast stretching
def contrast_stretching_mapping(x, minintensity, maxintensity):
	if(x >= minintensity and x <= maxintensity):
		return 20+((215.0/(maxintensity-minintensity))*(x-minintensity))
	if(x < minintensity):
		return (20.0/minintensity)*(x)
	return 255+((20/(255-maxintensity))*(x-255))

# function to perform contrast stretching on given image
def contrast_stretching(img, CDF):
	minintensity = 0
	maxintensity = 0
	for i in range(256):
		if(CDF[i] > 0.15):
			minintensity=i
			break
	for i in range(256):
		if(CDF[i]>0.95):
			maxintensity=i
			break
	#Define newimg
	newimg = np.zeros(img.shape,dtype=np.uint8)
	#mapping old image to newimage
	for j in range(newimg.shape[1]):
		for i in range(newimg.shape[0]):
			newimg[i][j] = int(contrast_stretching_mapping(img[i][j],minintensity,maxintensity))
	#Getting PIl object from np array
	newimg = Image.fromarray(newimg,mode="L")	
	newimg.save("contrast_stretching_01_09_"+image_name)

# function to perform gamma correction on given image with hyperparameter gamma value
def gamma_Correction(img,Gamma_value,image_name):
	# Define newimg
	newimg = np.zeros(img.shape,dtype=np.uint8)
	# mapping old image to newimage
	for j in range(newimg.shape[1]):
		for i in range(newimg.shape[0]):
			newimg[i][j] = int(255*((img[i][j]/255)**(Gamma_value)))
	#Getting PIl object from np array
	newimg = Image.fromarray(newimg,mode="L")	
	newimg.save("gamma_correction_"+image_name+"_"+str(Gamma_value)+".jpg")

# function to perform histogram equalization on given image
def Histogram_Equalization(img,image_name,CDF):
	Mapping = [0]*256
	#finding mapping by the formulae s=(L-1)*CDF[r] where L is 256 in this case
	for i in range(256):
		Mapping[i] = round(255*CDF[i])
	# define newimg
	newimg = np.zeros(img.shape,dtype=np.uint8)
	#mapping old image to newimage
	for j in range(newimg.shape[1]):
		for i in range(newimg.shape[0]):
			newimg[i][j] = Mapping[int(img[i][j])]
	#Getting PIl object from np array
	histogram2 = construct_histogram(newimg)
	plot_bar(histogram2,"intensity_values","frequency","Histogram for gray scale " + image_name,"Histogram_"+image_name)
	newimg=Image.fromarray(newimg,mode="L")	
	newimg.save("Histogram_equalization_Gray_scale_"+image_name)
	
# take 2D convolution of input image with given kernel
def convolution(img, kernel):
	# output image
	newImg = np.zeros(img.shape)
	mid_i = int(kernel.shape[0]/2)
	mid_j = int(kernel.shape[1]/2)
	for i in range(img.shape[0]):
		for j in range(img.shape[1]):
			conv = 0
			for a in range(kernel.shape[0]):
				for b in range(kernel.shape[1]):
					index_i = i + a - mid_i
					index_j = j + b - mid_j
					# boundary cases -- need to take care at edges
					if index_i >= 0 and index_j >= 0 and index_i < img.shape[0] and index_j < img.shape[1]:
						conv += kernel[a][b]*img[index_i][index_j]
			newImg[i][j] = int(conv)
	return np.array(newImg, dtype = np.uint8)

# apply unsharp masking on inout image.
# 1st step - blur image with gaussian kernel.
# 2nd step - new image = input_img + (input_img - blurred_img)*k
def Unsharp_masking(img,kernel,k_param,image_name):
	conv = convolution(img, kernel)
	Image.fromarray(conv, mode = "L").save("conv_" + image_name)
	ans = np.subtract(img, conv, dtype = float)
	ans *= k_param
	ans = np.add(img, ans, dtype = float)
	ans = np.array(ans, dtype = np.uint8)
	Image.fromarray(ans, mode = "L").save("ans_" + image_name)

# read image path via command line args
image_name=sys.argv[1]
im = Image.open(str(image_name))
# convert PIL object to numpy array
rgb = np.array(im)
# converting rgb to gray scale
r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
gray_scale_array = 0.2989 * r + 0.5870 * g + 0.1140 * b
# total number of pixels in grayscale image
total_pixels = gray_scale_array.shape[0]*gray_scale_array.shape[1]
histogram = construct_histogram(gray_scale_array)
CDF = find_CDF(histogram,total_pixels)
# # contrast_stretching
contrast_stretching(gray_scale_array, CDF)
# gamma_correction
a=[1.2, 1.4, 1.6, 1.8,2.2,2.6,3.0,3.4,3.8,4.0]
for i in a:
	gamma_Correction(gray_scale_array,1.0/i,image_name)
for i in a:
	gamma_Correction(gray_scale_array,i,image_name)
# Histogram_Equalization
Histogram_Equalization(gray_scale_array,image_name,CDF)
# Unsharp_masking
kernel=np.array([[1,1,1],[1,4,1],[1,1,1]], dtype = float)
kernel /= np.sum(kernel)
Unsharp_masking(gray_scale_array,kernel,0.5,image_name)

