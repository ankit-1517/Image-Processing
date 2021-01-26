import numpy as np 
import cv2

# ellipse or disk kernel
def diskKernel(size):
	return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*size - 1, 2*size - 1))

# perform image1 - image2
def intersect(image1 , image2):
	newimg = np.zeros(image1.shape , np.uint8)
	newimg[(image1 == 255) & (image2 == 255)] = 255
	return newimg	

# perform set difference
def difference(image1 , image2):
	newimg = np.zeros(image1.shape , np.uint8)
	newimg[(image1 == 255) & (image2 == 0)] = 255
	return newimg

# perform set difference
def Union(image1 , image2):
	newimg = np.zeros(image1.shape , np.uint8)
	newimg[(image1 == 255) | (image2 == 255)] = 255
	return newimg		

# Image operation using thresholding 
img = cv2.imread('Q3_1.bmp')

#Thresholding operation  
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
cv2.imwrite('imageth.jpg', thresh)

#Opening with the ellipse kernel having parameters ($7$ , $7$) to remove the small circle
opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, diskKernel(4), iterations = 13)
cv2.imwrite('opening.jpg', opening)

#Dilation with ellipse kernel having parameters ($5$ , $5$) to increase the size of the diminished big circle 
dilat = cv2.dilate(opening,diskKernel(4),iterations = 1)
cv2.imwrite('dilat.jpg', dilat)

#Intersection of constructed image with the original image to get finer details of large circle
inters = intersect(thresh,dilat)
cv2.imwrite('inters.jpg', inters)

#Image difference of original image with the image constructed so far to segment small circle
diff = difference(thresh , inters)
cv2.imwrite('diff.jpg', diff)

#Erosion of big circle with ellipse kernel having parameters ($5$ , $5$) to decrease size of big circle
erodebig = cv2.erode(inters,diskKernel(2),iterations = 1)
cv2.imwrite('erodebig.jpg', erodebig)
#Union of big circle with small circle to find the final image with separation line
union = Union(diff , erodebig)

cv2.imwrite('third.jpg', union)

