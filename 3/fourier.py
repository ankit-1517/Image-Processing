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

# rotate image by angle theta
def rotatecoordn(x,y,theta):
    x1=x*np.cos(2*np.pi-theta)-y*np.sin(2*np.pi-theta) # newx = xcos(theta)-ysin(theta)
    y1=x*np.sin(2*np.pi-theta)+y*np.cos(2*np.pi-theta) # newy = xsin(theta)+ycos(theta)
    return x1,y1 

# function to rotate image by angle theta
def rotate(img,theta):
    a=img.shape
    b=int(a[1]/2)
    c=int(a[0]/2)

    max1=2*int(abs((a[0]-c)*np.cos(theta)+(a[1]-b)*np.sin(theta)))
    max2=2*int(abs((a[0]-c)*np.sin(theta)+(a[1]-b)*np.cos(theta)))
    #print(max1+1)
    newimg = np.zeros((max1+1,max2+1),dtype=np.int8)
    d1=int((max1+1)/2)# new centre
    d2=int((max2+1)/2)# new centre
    for i in range(0,max1+1):
        for j in range(0,max1+1):
            xnew,ynew=rotatecoordn(j-d2,d1-i,theta)
            x=xnew+b
            y=c-ynew
            if(x<a[1] and y<a[0] and x>=0 and y>=0):
                newimg[i][j]=img[int(y)][int(x)]

    return newimg

#function to generate the spirit image
def generateImage(sizeX = 500, sizeY = 500, strips = 10, name = "strips.jpg"):
    img = np.zeros((sizeX, sizeY), dtype = np.uint8)
    width = int(sizeY/(2*strips))
    for j in range(500):
        white = int(j/width)%2
        for i in range(500):
            if white == 1:
                img[i][j] = 255
            else:
                img[i][j] = 0
    getImage(img).save(name)

# function to generate the white circle image
def generateWhiteCir(sizeX = 500, sizeY = 500, radius = 100, name = "whiteCir.jpg"):
    img = np.zeros((sizeX, sizeY), dtype = np.uint8)
    for j in range(500):
        for i in range(500):
            if ((i-(sizeY/2))**2)+((j-(sizeX/2))**2) <= radius*radius:
                img[i][j] = 255
            else:
                img[i][j] = 0
    getImage(img).save(name)

#function to generate the black circle image
def generateBlackCir(sizeX = 500, sizeY = 500, radius = 100, name = "balckCir.jpg"):
    img = np.zeros((sizeX, sizeY), dtype = np.uint8)
    for j in range(500):
        for i in range(500):
            if ((i-(sizeY/2))**2)+((j-(sizeX/2))**2) <= radius*radius:
                img[i][j] = 0
            else:
                img[i][j] = 255
    getImage(img).save(name)

# function to find the fourier transform of image
def fourier_n(img):
    # # preprocessing the image for centering the zero frequency
    # img1 = np.zeros(img.shape , dtype = np.float32)
    # for i in range(img.shape[0]):
    #     for j in range(img.shape[1]):
    #         img1[i,j] = (-1**(i+j))*img[i,j]

    N = img.shape[0]
    exp1 = np.zeros((N,N) , dtype=np.complex)
    const = -(2*math.pi)/N

    for y in range(N):
        for v in range(N):
            exp1[y,v] = cmath.exp(complex(0, const*y*v))
    
    temp = img.dot(exp1) 
    final_img = (exp1.transpose()).dot(temp)
    return final_img

# function to find the inverse fourier transform of image
def inverse_fourier_n(img):
    N = img.shape[0]
    exp1 = np.zeros((N,N) , dtype=np.complex)
    const = (2*math.pi)/N

    for y in range(N):
        for v in range(N):
            exp1[y,v] = cmath.exp(complex(0, const*y*v))
    
    temp = img.dot(exp1) 
    final_img = (exp1.transpose()).dot(temp)
    final_img = (1.0/(N*N))*final_img
    final_img = abs(final_img)

    
    # for i in range(img.shape[0]):
    #     for j in range(img.shape[1]):
    #         final_img[i,j] = (-1**(i+j))*final_img[i,j]

    final_img1 = np.zeros((N,N) , dtype=np.uint8)
    final_img1 = final_img.astype(np.uint8)
    
    return final_img1


# generateImage()
img = readImage('strips.jpg')
# temp = getImage(img)
# temp.save('blurred_image.jpg')

original=np.fft.fft2(img)

#checking the correct fourier of image
plt.plot(abs(original))
plt.savefig('correct_fourier.jpg')

# fourier of image
F = fourier_n(img)

#function to center the  zero frequency component
F = np.fft.fftshift(F)

#for plotting the real part of fft
plt.clf()
plt.plot(abs(F.real))
getImage(abs(F.real).astype(np.uint8)).save("F_real.jpg")
plt.savefig('My_fourier_real.jpg')


#for plotting the imaginary party of fft
plt.clf()
plt.plot(abs(F.imag))
plt.imsave('test.png',abs(F.imag).astype(np.uint8),cmap = 'gray' )
getImage(abs(F.imag).astype(np.uint8)).save("F_imag.jpg")
plt.savefig('My_fourier_imaginary.jpg')

#rotating image

newimg = rotate(img,np.pi/4)
#newimg = np.array(getImage(img).rotate(45))
# newimg = getArray(rotated)
saveImage(newimg,"L","Rotated_image.jpg")
fourier_rotated_image=fourier_n(newimg)
fourier_rotated_image = np.fft.fftshift(fourier_rotated_image)

#for plotting the real part of fft
plt.clf()
plt.plot(abs(fourier_rotated_image.real))
plt.imsave('rotated_real.png',abs(fourier_rotated_image.real).astype(np.uint8),cmap = 'gray' )
plt.savefig('rotated_fourier_real.jpg')

#for plotting the imaginary party of fft
plt.clf()
plt.plot(abs(fourier_rotated_image.imag))
plt.imsave('rotated_imag.png',abs(fourier_rotated_image.imag).astype(np.uint8),cmap = 'gray' )
plt.savefig('rotated_fourier_imaginary.jpg')

#generate White Circle image
generateWhiteCir()
#generate Black Circle
generateBlackCir()
#Part three
img = readImage('whiteCir.jpg')
Output_3 = img*F
final_img = inverse_fourier_n(Output_3)
getImage(final_img).save("part3_3.jpg")
#Part four 
img = readImage('balckCir.jpg')
Output_4 = img*F
final_img = inverse_fourier_n(Output_4)
getImage(final_img).save("part4_4.jpg")