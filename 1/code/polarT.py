# importing PIL -- useful in reading/writing/viewing images
from PIL import Image 
# importing math -- useful in tan inverse, cos(theta) etc
import math
# importing numpy -- useful in matrix operations
import numpy as np
# importing sys -- useful in command line arguments
import sys

# function to read image at a given path
def readImage(path):
    return np.asarray(Image.open(path))

# convert np.array to Image.object
def getImage(img):
    return Image.fromarray(img, mode = 'RGB')

# convert (r, theta) to polar coordinates (x, y)
def polarToCart(r, theta, center_x, center_y):
    x = int(r*math.sin(math.radians(theta)) + center_y)
    y = int(r*math.cos(math.radians(theta)) + center_x)
    return x, y

def polarTransform(img):
    # find "radius" of original image
    a, b = img.shape[1]/2, img.shape[0]/2
    r = math.sqrt(a**2+b**2)
    # define output image array
    img2 = np.zeros((int(r), 360, 3), dtype = np.int8)
    for r in range(img2.shape[0]):
        for theta in range(img2.shape[1]):
            # inverse-map new image coordinates to avoid black-spots in output image
            x, y = polarToCart(r, theta, a, b)
            if x>= 0 and y>= 0 and x<= img.shape[1]-1 and y<= img.shape[0]-1:
                img2[r][theta] = img[int(x)][int(y)]
    return img2

# input image
try:
    # take input file name from command line
    img = str(sys.argv[1])
except:
    print("Input file not given. Run python3 \"<file_name>.py <imageName>\".")
    exit(1)
# obtaining file name and file extension. Helful in generalising our code.
# Won't have to change anything in the code to run on any image.
# l[0] contains image name and l[1] contains image extension
l = img.split('.')
# read input image file -- in np.array format currently
img = readImage(img)
# apply polar transformation
img = polarTransform(img)
# get PIL image from np.array
img = getImage(img)
# save image will given name
img.save(l[0] + '_polar.' +l[1])
# can also use img.show() to show image
