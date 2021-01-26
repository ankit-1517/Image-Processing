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
#plt.ion()
def readImage(path):
    return np.asarray(Image.open(path))

# convert np.array to Image.object
def getImage(img):
    return Image.fromarray(img, mode = 'RGB')

def getArray(img):
    return np.asarray(img)
#function to construct the histogram
def construct_histogram(np_im):
	histogram=[0]*256
	for k in range(dimen[2]):
		for j in range(dimen[1]):
			for i in range(dimen[0]):
				histogram[np_im[i][j][k]]=histogram[np_im[i][j][k]]+1
	return histogram	
#function to plot_bar graph
def plot_bar(y_values,xlabel,ylabel,title,name):
	B=range(256)
	plt.bar(B,y_values, color='g')
	plt.xlabel(xlabel)
	plt.ylabel(ylabel)
	plt.title(title)
	plt.savefig(name)
	plt.close()
	
#function to find the mean 
def find_mean(histogram):
	mean=0
	for i in range(256):
		mean=mean+(i*histogram[i])
	return mean
#function to find standard_deviation
def find_standard_deviation(histogram):
	mean=find_mean(histogram)
	variance=0
	for i in range(256):
		variance=variance+(((i-mean)**2)*histogram[i])
	return np.sqrt(variance)
#function to find Energy
def find_Energy(histogram):
	energy=0
	for i in range(256):
		energy=energy+(histogram[i]**2)
	return energy
#function to find entropy
def find_Entropy(histogram):
	entropy =0
	for i in range(256):
		if(histogram[i]!=0):
			entropy=entropy - histogram[i]*np.log2(histogram[i])
	return entropy
#function to find skewness of histogram
def find_skewness(histogram):
	sigma=find_standard_deviation(histogram)
	mean=find_mean(histogram)
	Third_moment=0
	for i in range(256):
		Third_moment=Third_moment+(((i-mean)**3)*histogram[i])
	skewness = (Third_moment*1.0)/(sigma**3)
	return skewness
#function to find kurtosis of histogram
def find_Kurtosis(histogram):
	sigma=find_standard_deviation(histogram)
	mean=find_mean(histogram)
	fourth_moment=0
	for i in range(256):
		fourth_moment=fourth_moment+(((i-mean)**4)*histogram[i])
	kurtosis = (fourth_moment*1.0)/(sigma**4)
	return kurtosis
#function to find CDF
def find_CDF(histogram,name):
	CDF=[0]*256
	temp=0
	for i in range(256):
		temp=temp+histogram[i]
		CDF[i]=temp
	CDF_normalised= [i*255 for i in CDF]
	#print(CDF)
 	#plot CDF of the histogram unormalised
	plot_bar(CDF,'Intensity Values','Cummulative Probability','CDF','CDF_for_unnormalised'+a)
	plot_bar(CDF_normalised,'Intensity Values','Cummulative value (Normalised)','CDF','CDF_for_normalised'+a)
a=sys.argv[1]
im = Image.open(str(a))
np_im = np.array(im)
dimen=np_im.shape
total_pixels=dimen[0]*dimen[1]*dimen[2]
#Part1
histogram=construct_histogram(np_im)
histogram1=[i*1.0/total_pixels for i in histogram]
#Part1.1
plot_bar(histogram,'Intensity Values','frequency','Histogram','Histogram_for_'+a)
#Part1.2
#Subpart a
mean=find_mean(histogram1)
print("mean:"+str(mean))

#Subpart b
standard_deviation=find_standard_deviation(histogram1)
print("standard_deviation:"+str(standard_deviation))

#Subpart c
energy= find_Energy(histogram1)
print("Energy:"+str(energy))

#Subpart d
entropy= find_Entropy(histogram1)
print("Entropy:"+str(entropy))

#Subpart e
skewness=find_skewness(histogram1)
print("skewness:"+str(skewness))

#Subpart f
kurtosis =find_Kurtosis(histogram1)
print("kurtosis:"+str(kurtosis))

#Subpart 1.3
find_CDF(histogram1,a)
