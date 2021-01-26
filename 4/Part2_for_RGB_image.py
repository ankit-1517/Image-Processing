# importing PIL -- useful in reading/writing/viewing images
from PIL import Image
from PIL import ImageFilter
# importing math -- useful in tan inverse, cos(theta) etc
import math
import cmath
# importing numpy -- useful in matrix operations
import numpy as np
# importing sys -- useful in command line arguments
import sys
#importing matplotlib in pythn
import matplotlib.pyplot as plt
import scipy.misc
#importing scipy in python
from scipy import ndimage
# function to read image at a given path
def readImage(path):
    return np.asarray(Image.open(path))

# convert np.array to Image.object
def getImage(img):
    return Image.fromarray(img, mode = 'RGB')

# convert PIL image to np array
def getArray(img):
    return np.asarray(img)

# fuunction to save image
def saveImage(newimg,mode1,image_name):
    newimg=Image.fromarray(newimg,mode=mode1)   
    newimg.save(image_name)
#function to convolve image with sobel operator
def sobel_operator_to_give_energy(im):
	im = im.astype('int32')
	dx = ndimage.sobel(im, 0)  # horizontal derivative
	dy = ndimage.sobel(im, 1)  # vertical derivative
	# mag = np.hypot(dx, dy)  # magnitude
	mag = np.abs(dx) + np.abs(dy)
	return mag
#function to find the minimum horizontal_seam:
def remove_horizontal_seam(sobel_img,original_img):
	newimg = np.zeros((original_img.shape[0]-1,original_img.shape[1],3), dtype = np.uint8)
	M = np.zeros(sobel_img.shape)

	#loop to fill M matrix 
	for j in range(sobel_img.shape[1]):
		for i in range(sobel_img.shape[0]):
			min1 = 0
			if(j-1 >= 0):
				min1 = M[i,j-1]
			if(i-1 >= 0 and j-1 >= 0):
				if( min1 > M[i-1,j-1] ):
					min1 = M[i-1,j-1]
			if( j-1 >= 0 and i+1 < sobel_img.shape[0] ):
				if(min1 > M[i+1,j-1]):
					min1 = M[i+1,j-1]
			M[i,j] = sobel_img[i,j] + min1
	
	index = np.argmin( M[:,sobel_img.shape[1]-1] )
	i = index
	j = sobel_img.shape[1]-1

	while( j >= 0 ):
		for k in range(i):
			newimg[k,j] = original_img[k,j]

		for k in range(i,original_img.shape[0]-1):
			newimg[k,j] = original_img[k+1,j]

		if(j>0):
			if(i > 0 and i+1< sobel_img.shape[0]):
				if(M[i-1][j-1] < M[i][j-1] and M[i-1][j-1] < M[i+1][j-1]):
					i = i - 1	
				elif(M[i+1][j-1] < M[i][j-1] and M[i+1][j-1] < M[i-1][j-1]):
					i = i + 1
			elif(i > 0):
				if(M[i-1][j-1] < M[i][j-1]):
					i = i - 1
			elif(i+1 < sobel_img.shape[1]):
				if(M[i+1][j-1] < M[i][j-1]):
					i = i + 1
		j = j-1
	return newimg , M[index,sobel_img.shape[1]-1,]

#function to remove the minimum vertical seam:
def remove_vertical_seam(sobel_img,original_img):
	newimg = np.zeros((original_img.shape[0],original_img.shape[1]-1,3), dtype = np.uint8)
	M = np.zeros(sobel_img.shape)

	#loop to fill M matrix 
	for i in range(sobel_img.shape[0]):
		for j in range(sobel_img.shape[1]):
			min1 = 0
			if(i-1 >= 0):
				min1 = M[i-1,j]
			if(i-1 >= 0 and j-1 >= 0):
				if( min1 > M[i-1,j-1] ):
					min1 = M[i-1,j-1]
			if( i-1 >= 0 and j+1 < sobel_img.shape[1] ):
				if(min1 > M[i-1,j+1]):
					min1 = M[i-1,j+1]
			M[i,j] = sobel_img[i,j] + min1
	#print(sobel_img)
	#print(M)
	# Now the minimum energy seam will end at the minimum energy position in last row
	
	index = np.argmin( M[sobel_img.shape[0]-1,:] )
	i = sobel_img.shape[0]-1
	j = index

	while( i >= 0 ):
		for k in range(j):
			newimg[i,k] = original_img[i,k]

		for k in range(j,original_img.shape[1]-1):
			newimg[i,k] = original_img[i,k+1]

		if(i>0):
			if(j > 0 and j+1< sobel_img.shape[1]):
				if(M[i-1][j-1] < M[i-1][j] and M[i-1][j-1] < M[i-1][j+1]):
					j = j - 1	
				elif(M[i-1][j+1] < M[i-1][j] and M[i-1][j+1] < M[i-1][j-1]):
					j = j + 1
			elif(j > 0):
				if(M[i-1][j-1] < M[i-1][j]):
					j = j-1
			elif(j+1 < sobel_img.shape[1]):
				if(M[i-1][j+1] < M[i-1][j]):
					j = j+1			
		i = i-1
	
	return newimg , M[sobel_img.shape[0]-1,index]
def seam_carving_compression(img , row_dec , col_dec):
	#remove row_dec horizontal and col_dec vertical seam
	newimg = img
	for i in range(0,col_dec):
		mag = sobel_operator_to_give_energy(newimg)
		mag1 = np.sum(mag,axis=2) 
		newimg , energy = remove_vertical_seam(mag1,newimg)
	#saveImage(newimg,"RGB",'after_vertical_seam_removal.jpg')

	for i in range(0,row_dec):
		mag = sobel_operator_to_give_energy(newimg)
		mag1 = np.sum(mag,axis=2) 
		newimg , energy = remove_horizontal_seam(mag1,newimg)
	print(newimg.shape)
	saveImage(newimg,"RGB",'seam_coloured'+str(row_dec)+"_"+str(col_dec)+".jpg")
L = 0

# recurrsive function to find optimal_seam_carving_compression
def optimal_seam_carving_compression(img , row_dec, col_dec):
	global L
	if(row_dec == 0 and col_dec == 0):
		return 0
	
	mag = sobel_operator_to_give_energy(img)
	energy1 = 0
	energy2 = 0
	energy3 = 0
	energy4 = 0
	newimg2 = np.zeros((img.shape[0]-row_dec,img.shape[1]-col_dec), dtype = np.uint8) 
	newimg3 = np.zeros((img.shape[0]-row_dec,img.shape[1]-col_dec), dtype = np.uint8)
	
	if(row_dec>0):
		newimg , energy1 = remove_horizontal_seam(mag,img)
		if(L[row_dec-1][col_dec]>0):
			energy2 = L[row_dec-1][col_dec]
		else:
			energy2 = optimal_seam_carving_compression(newimg , row_dec-1, col_dec)
			L[row_dec-1][col_dec] = energy2
			#saveImage(newimg2,"L","optimal_seam"+str(row_dec-1)+"_"+str(col_dec)+"_"+".jpg")
	if(col_dec>0):
		newimg , energy3 = remove_vertical_seam(mag,img)
		if(L[row_dec][col_dec-1]>0):
			energy4 = L[row_dec][col_dec-1]
		else:
			energy4 = optimal_seam_carving_compression(newimg , row_dec, col_dec-1)
			L[row_dec][col_dec-1] = energy4
		#saveImage(newimg3,"L","optimal_seam"+str(row_dec)+"_"+str(col_dec-1)+"_"+".jpg")
	if(row_dec == 0):
		return  energy3+energy4

	if(col_dec==0):
		return energy1+energy2

	if(energy1+energy2 < energy3+energy4):
		return energy1+energy2

	return energy3+energy4

def optimal_seam_carving_compression_generator(img , row_dec, col_dec):
	global L
	
	L = np.zeros((row_dec+1,col_dec+1))
	energy = optimal_seam_carving_compression(img,row_dec,col_dec)
	L[row_dec][col_dec] = energy
	newimg = img
	print(L)

	while(row_dec>0 or col_dec>0):
		if(row_dec==0):
			col_dec = col_dec-1
			mag = sobel_operator_to_give_energy(newimg)
			newimg , energy = remove_vertical_seam(mag,newimg)
		elif(col_dec==0):
			row_dec = row_dec-1
			mag = sobel_operator_to_give_energy(newimg)
			newimg ,  energy  = remove_horizontal_seam(mag,newimg)
		else:
			if(L[row_dec-1][col_dec]<L[row_dec][col_dec-1]):
				row_dec = row_dec-1
				mag = sobel_operator_to_give_energy(newimg)
				newimg ,  energy  = remove_horizontal_seam(mag,newimg)
			else:
				col_dec = col_dec-1
				mag = sobel_operator_to_give_energy(newimg)
				newimg ,  energy  = remove_vertical_seam(mag,newimg)

	return newimg , energy


img = readImage(sys.argv[1])
saveImage(img,"RGB","img.jpg")
 #for black and white
seam_carving_compression(img,50,50)



