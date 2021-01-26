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
#importing matplotlib in python
import matplotlib.pyplot as plt

# function to read image at a given path
def readImage(path):
    return np.asarray(Image.open(path))

# convert np.array to Image.object
def getImage(img):
    return Image.fromarray(img, mode = 'L')

# convert PIL image to np array
def getArray(img):
    return np.asarray(img)

# fuunction to save image
def saveImage(newimg,mode1,image_name):
    newimg=Image.fromarray(newimg,mode=mode1)   
    newimg.save(image_name)

def f():
    img = readImage('Rotated_image.jpg')
    newX = int(img.shape[0]/2)
    newY = int(img.shape[1]/2)
    newImg = np.zeros((newX, newY), dtype=np.uint8)
    newX = int(newX/2)
    newY = int(newY/2)
    for i in range(newImg.shape[0]):
        for j in range(newImg.shape[1]):
            newImg[i][j] = img[newX+i][newY+j]
    saveImage(newImg, 'L', 'my_rot.jpg')

img = np.zeros((500, 500), dtype=np.int8)
for i in range(500):
    for j in range(500):
        if (i-j)%100>50:
            img[i][j] = 255
saveImage(img, 'L', 'my_rot2.jpg')
