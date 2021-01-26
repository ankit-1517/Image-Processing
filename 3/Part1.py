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
# function to read image at a given path

def readImage(path):
    return np.asarray(Image.open(path))

# convert np.array to Image.object
def getImage(img):
    return Image.fromarray(img, mode = 'RGB')

def getArray(img):
    return np.asarray(img)

def saveImage(newimg,mode1,image_name):
	newimg=Image.fromarray(newimg,mode=mode1)	
	newimg.save(image_name)

def apply_difference_equation(image):
	
	outputimage = np.zeros(image.shape, dtype=np.float32)
	image = image.astype(np.float32)

	for j in range(0,image.shape[1]):
		
		for i in range(0,image.shape[0]):
			
			temp = 0.01*image[i][j][:] 
			
			if(i-1 >= 0):
				temp +=  0.9 * outputimage[i-1][j][:]
			
			if(j-1 >= 0):
				temp +=  0.9 * outputimage[i][j-1][:]
			
			if(i-1 >= 0 and j-1 >= 0):
				temp -=  0.81*outputimage[i-1][j-1][:]
		
			outputimage[i][j][:] = temp

	return outputimage.astype(np.uint8)

def apply_difference_equation_bw(image):
	
	outputimage = np.zeros(image.shape , dtype=np.float32)
	image = image.astype(np.float32)
	for j in range(0,image.shape[1]):
		
		for i in range(0,image.shape[0]):
			
			temp = 0.01*image[i][j]
			
			if(i-1 >= 0):
				temp +=  0.9 * outputimage[i-1][j]
			
			if(j-1 >= 0):
				temp +=  0.9 * outputimage[i][j-1]
			
			if(i-1 >= 0 and j-1 >= 0):
				temp -=  0.81*outputimage[i-1][j-1]
			
			outputimage[i][j] = temp

	final_output_image = outputimage.astype(np.uint8)
	return final_output_image
# take 2D convolution of input image with given kernel
def convolution(img, kernel):
	img = img.astype(np.float32)
	kernel = kernel.astype(np.float32)
	# output image
	newImg = np.zeros(img.shape , dtype = np.float32)
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
			# newImg[i][j] = conv.astype(int)
			newImg[i][j] = conv
	# return np.array(newImg, dtype = np.uint8)
	return newImg
# apply unsharp masking on inout image.
# 1st step - blur image with gaussian kernel.
# 2nd step - new image = input_img + (input_img - blurred_img)*k
def Unsharp_masking(img,kernel,k_param,image_name):
	img = img.astype(np.float32)
	conv = convolution(img, kernel)
	Image.fromarray(conv.astype(np.uint8), mode = "RGB").save("conv_" + image_name)
	ans = np.subtract(img, conv, dtype = float)
	ans *= k_param
	ans = np.add(img, ans, dtype = float)
	ans = np.array(ans, dtype = np.uint8)
	plt.imsave('test.png',ans.astype(np.uint8))
	Image.fromarray(ans, mode = "RGB").save("ans_" + image_name)

#Part 1.1.1 
input_image = np.zeros((256,256), dtype=np.uint8)

# Asuming  index 127 is correspoind to 1 start
input_image[126][126] = 255
saveImage(input_image,'L',"Part1input.jpg")
input_image=input_image
#print(input_image[126:,126:])
modified_image=apply_difference_equation_bw(input_image)
#print(modified_image[126:,126:])
saveImage((modified_image*100).astype(np.uint8),'L',"Part1a.jpg")

# #Part1.1.2
image1 = getArray(readImage(sys.argv[1]))
#saveImage(image1,"RGB","Part1binput.jpg")
modified_image=apply_difference_equation(image1)
#print(modified_image)
saveImage(modified_image.astype(np.uint8),"RGB","Part1b.jpg")

# Part 2.1.1
# Unsharp_masking
kernel=np.ones((5,5) , dtype = float)
kernel /= np.sum(kernel)

Unsharp_masking(image1,kernel,0.2,"Unsharp_masking02.jpg")
Unsharp_masking(image1,kernel,0.8,"Unsharp_masking08.jpg")
Unsharp_masking(image1,kernel,1.5,"Unsharp_masking15.jpg")