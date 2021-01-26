# importing PIL -- useful in reading/writing/viewing images
from PIL import Image 
# importing numpy -- useful in matrix operations
import numpy as np
import sys
import matplotlib.pyplot as plt
import pywt
import pywt.data

# function to read image at a given path
def readImage(path):
    return np.asarray(Image.open(path))

def saveImg(img, name, colorMap = plt.cm.gray):
    plt.imsave(name + ".jpg", img, cmap=colorMap)

def MSE(image1 , image2):
	prod = 1
	for i in image1.shape:
		prod *= i 
	temp = image1.astype(np.float64) - image2.astype(np.float64)
	temp = temp*temp
	MSE = np.sum(temp)*1.0/prod
	return MSE

def waveletDecomposition(img, name):
    titles = ['LL', 'LH', 'HL', 'HH', 'original']
    coeffs2 = pywt.dwt2(img, 'bior1.3')
    LL, (LH, HL, HH) = coeffs2
    origin = pywt.idwt2(coeffs2,'bior1.3')
    l = [LL, LH, HL, HH, origin]
    for x, a in enumerate(l):
        saveImg(a, name + "_" + titles[x])
    return l

l = str(sys.argv[1])
lSplit = l.split('.')
img = readImage(l)
print(img.shape)
l1 = waveletDecomposition(img, lSplit[0])
print(MSE(img, l1[4]))
l2 = waveletDecomposition(l1[0], "LL")
originHalf = pywt.idwt2((l2[0], (l2[1], l2[2], l2[3])),'bior1.3')
originalFull = pywt.idwt2((originHalf, (l1[1], l1[2], l1[3])), 'bior1.3')
print(MSE(originalFull, img))
saveImg(originalFull, lSplit[0] + 'Final')
