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
def getImage(img,mode1):
    return Image.fromarray(img, mode = mode1 )

def getArray(img):
    return np.asarray(img)

def saveImage(newimg,mode1,image_name):
	newimg=Image.fromarray(newimg,mode=mode1)	
	newimg.save(image_name)

# function to find dct of size 
def find_matrix_for_dct(size):
	mat = np.zeros((size,size), dtype=float)
	for v in range(0,size):
		for u in range(0,size):
			if(u==0):
				mat[u][v] = 1.0/np.sqrt(size)
			else:
				mat[u][v] = (np.sqrt(2.0)/np.sqrt(size))*np.cos(((2*v+1)*np.pi*u)/(2*size)) 
	return mat

#function to find dct of image
def find_dct_image(image,cmat,size):
	
	newimg=np.zeros(image.shape , dtype = float)
	image = image - 128

	i = 0
	while( i < image.shape[0] ):
		j = 0
		while( j < image.shape[1] ):
			temp = image[i:i+size,j:j+size] 
			newimg[i:i+size,j:j+size] = (cmat.dot(temp)).dot(cmat.transpose())	 
			j=j+size
		i=i+size

	return newimg

#function to find the inverse cosine transform of image	
def inverse_dct_image(image,cmat,size):
	newimg=np.zeros(image.shape , dtype = float)

	i = 0
	while( i < image.shape[0] ):
		j = 0
		while( j < image.shape[1] ):
			temp = image[i:i+size,j:j+size]
			temp2 = (np.rint((cmat.transpose().dot(temp)).dot(cmat)))+128	
			newimg[i:i+size,j:j+size] = temp2 
			j=j+size
		i=i+size

	newimg = newimg.astype(np.uint8)
	return newimg

#function to select cofficients from  image
def selecting_cofficent(max_number_coff , dct_img,size):
	# initialising new image map1 and average_energy
	newimg= np.zeros(dct_img.shape, dtype = float)
	map1 = np.zeros((size,size) , dtype = float)
	average_energy = np.zeros((size,size) ,dtype = float)
	i=0
	# loop to find the average energy of each pixel
	while( i < dct_img.shape[0] ):
		j=0
		while( j < dct_img.shape[1] ):
			average_energy = average_energy + dct_img[i:i+size,j:j+size]
			j=j+size
		i=i+size

	# finding average energy and sorting array to find the thershold 
	average_energy = abs((average_energy*1.0)/((dct_img.shape[0]*dct_img.shape[1])/(size*size)))
	print("average energy")
	print(average_energy)
	plt.clf()
	plt.imsave('average_energy_'+str(size)+'_'+str(max_number_coff)+'.jpeg',abs(average_energy).astype(np.uint8),cmap = 'gray' )
	temp = average_energy.flatten()
	temp.sort()
	total_zero_cofficient = 0
	
	# loop to find the map
	for i in range(size):
		for j in range(size):

			if(average_energy[i][j] >= temp[-max_number_coff]):
				map1[i,j] = 1
			else :
				map1[i,j] = 0
				total_zero_cofficient += 1
	print("map :")
	print(map1)
	# total number of zero cofficient in the matrix
	total_zero_cofficient = (total_zero_cofficient * (dct_img.shape[0]*dct_img.shape[1])) / (size*size)
	#  image after thresholding
	i=0
	while( i < dct_img.shape[0] ):
		j=0
		while( j < dct_img.shape[1] ):

			for t in range(i,i+size):
				for s in range(j,j+size):

					if(map1[t-i,s-j]==1):
						newimg[t,s] = dct_img[t,s]
					else:
						newimg[t,s] = 0
			
			j=j+size
		i=i+size
	#print(total_zero_cofficient)
	return newimg , map1 , average_energy , total_zero_cofficient

def for_colour_image(max_coff,size):
	print(" ")
	print("size: "+str(size)+" max_coff: "+str(max_coff))
	image = getImage(readImage(sys.argv[1]),'RGB')
	image = getArray(image)
	image = image.astype(float)
	cmat = find_matrix_for_dct(size)
	uncompressed_img = np.zeros(image.shape , dtype = np.uint8)
	#print(image.shape[2])
	total_zero_cofficient = 0
	for k in range(0,image.shape[2]):
		dct_img = find_dct_image(image[:,:,k],cmat,size)
		dct_compressed_img , map1 , average_energy , total_zero_cofficient1 = selecting_cofficent(max_coff , dct_img,size)	
		total_zero_cofficient += total_zero_cofficient1 
		temp = inverse_dct_image(dct_compressed_img,cmat,size)
		uncompressed_img[:,:,k]=temp
	saveImage(uncompressed_img,'RGB',"dct_compressed"+str(max_coff)+"_"+str(size)+"_"+sys.argv[1])
	MSE = image.astype(float) - uncompressed_img.astype(float)
	MSE = np.sum(MSE*MSE)/(image.shape[0]*image.shape[1])
	print("MSE/pixel :"+str(MSE))
	print("total coefficients which are converted to zero: "+str(total_zero_cofficient))

def for_blackandwhite_image(max_coff,size):
	# getting the image as numpy  array
	image = getImage(readImage(sys.argv[1]),'L')
	image = getArray(image)
	print(image.shape)
	image = image.astype(float)
	# finding the matrix for dct
	cmat = find_matrix_for_dct(size)
	#finding dct of image
	dct_img = find_dct_image(image,cmat,size)
	dct_compressed_img , map1 , average_energy , total_zero_cofficient = selecting_cofficent(max_coff , dct_img,size)	
	uncompressed_img = inverse_dct_image(dct_compressed_img,cmat,size)
	saveImage(uncompressed_img,'L',"zdct_compressed"+str(max_coff)+"_"+str(size)+"_"+"zfinal.jpg")
	MSE = image.astype(float) - uncompressed_img.astype(float)
	MSE = np.sum(MSE*MSE)/(image.shape[0]*image.shape[1])
	print("MSE : "+ str(MSE))

for_blackandwhite_image(10,8)
for_blackandwhite_image(5,8)
for_blackandwhite_image(40,16)
for_blackandwhite_image(10,16)
for_blackandwhite_image(150,16)

# for_colour_image(10,8)
# for_colour_image(5,8)
# for_colour_image(40,16)
# for_colour_image(150,16)
# for_colour_image(10,16)
# for_colour_image(30,8)
