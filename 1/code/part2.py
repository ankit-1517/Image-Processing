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

# convert PIL image to np array
def getArray(img):
    return np.asarray(img)

# function to flip image
# axis = 0 : mirror image wrt x axis (vertical flip), axis = 1 : mirror image wrt y axis (horizontal flip)
def flip(img, axis = 1):
    # define output image array. since it's flip, dimensions remain the same
    img2 = np.zeros(img.shape, dtype = np.int8)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if axis == 0:
                # Vertical flip
                img2[i][j] = img[img.shape[0]-1-i][j]
                img2[img.shape[0]-1-i][j] = img[i][j]
            else:
                # Horizontal flip
                img2[i][j] = img[i][img.shape[1]-1-j]
                img2[i][img.shape[1]-1-j] = img[i][j]
    return img2

# function to translate image
# t_x and t_y are translations in x and y direction respectively 
def translate(img, t_x, t_y):
    # Since fractional translations won't even count, convert them down to integers
    t_x, t_y = int(t_x), int(t_y)
    # define output image array
    img2 = np.zeros((img.shape[0] + abs(t_y), img.shape[1] + abs(t_x), img.shape[2]), dtype=np.int8)
    # transformation matrix
    M = np.array(((1, 0, t_x), (0, 1, t_y), (0, 0, 1)))
    M = np.linalg.inv(M)
    # compute old and new origin
    c_x, c_xn, c_y, c_yn = img.shape[1]/2, img2.shape[1]/2, img.shape[0]/2, img2.shape[0]/2
    # compute offset, since there will be change of origin upon translation
    offset = [c_xn - c_x, c_yn - c_y]
    for i in range(img2.shape[0]):
        for j in range(img2.shape[1]):
            # find inverse-mapped coordinates, currently in new origin
            arr = np.dot(M, np.array((j - c_xn + offset[0], c_yn - i + offset[1], 1)).T)
            # convert from new origin to previous origin (because need intensity values from previous/input image)
            temp = arr[0]
            arr[0] = c_y - arr[1]
            arr[1] = c_x + temp
            # put only if lie in this range, the new image should be black in other places, to account for the transformation
            if arr[0] >= 0 and arr[0] < img.shape[0] and arr[1] >= 0 and arr[1] < img.shape[1]:
                img2[i][j] = img[int(arr[0])][int(arr[1])]
    # return new (transformed) image
    return img2

# function to rotate image
# theta is taken positive in anti-clockwise direction
def rotate(img, theta):
    # take origin to be center
    center_x, center_y = img.shape[1]/2, img.shape[0]/2
    # find radius of "circle" along which the image will be rotated
    r = math.sqrt(center_x**2 + center_y**2)    
    # transformation will be same for theta and theta%360
    theta %= 360
    # find new origin of the image
    if theta == 0 or theta == 180:
        # for 0 or 180 degrees, there is no change in dimensions
        center_x_n, center_y_n = center_x, center_y
    elif theta == 90 or theta == 270:
        # for 90 or 270 degrees, the dimesnions will simply be swapped
        center_x_n, center_y_n = center_y, center_x
    else:
        # otherwise, new dimensions can be found using this formula
        center_x_n, center_y_n = r*math.cos(math.radians(45-(theta%90))), r*math.sin(math.radians(45+(theta%90)))
    # define output image array
    img2 = np.zeros((int(2*center_y_n), int(2*center_x_n), img.shape[2]), dtype=np.int8)
    theta = math.radians(theta)
    # rotation matrix
    M = np.array(((math.cos(theta), -math.sin(theta), 0), 
        (math.sin(theta), math.cos(theta), 0), 
        (0, 0, 1)))
    # take inverse -- useful in inverse-mapping coordinates
    M = np.linalg.inv(M)
    for i in range(img2.shape[0]):
        for j in range(img2.shape[1]):
            # find inverse-mapped coordinates, currently in new origin
            arr = np.dot(M, np.array((j - center_x_n, center_y_n - i, 1)).T)
            # convert from new origin to previous origin (because need intensity values from previous/input image)
            temp = arr[0]
            arr[0] = center_y - arr[1]
            arr[1] = center_x + temp
            # put only if lie in this range, the new image should be black in other places, to account for the transformation
            if arr[0] >= 0 and arr[0] < img.shape[0] and arr[1] >= 0 and arr[1] < img.shape[1]:
                img2[i][j] = img[int(arr[0])][int(arr[1])]
    # return new (transformed) image
    return img2

# function to scale image
# s_x and s_y are scale factors in x and y directions respectively
def scale(img, s_x, s_y):
    # define output image array, multiply by scale factors to obtain new shape
    img2 = np.zeros((int(img.shape[0]*s_x), int(img.shape[1]*s_y), img.shape[2]), dtype = np.int8)
    for i in range(img2.shape[0]):
        for j in range(img2.shape[1]):
            # inverse-map coordinates to prevent black-spots
            img2[i][j] = img[int(i/s_x)][int(j/s_y)]
    # return new (transformed) image
    return img2

# shear	transformation in x - direction
# k is transformation scale 
def shear(img, k):
    # define output image array
    # will be elongated along x-axis (shape[1])
    img2 = np.zeros((img.shape[0], int(img.shape[0]*k + img.shape[1]), img.shape[2]), dtype=np.int8)
    # origin will not shift in y-direction, no need to consider thar
    c_x, c_xn, c_y = img.shape[1]/2, img2.shape[1]/2, img.shape[0]/2
    # shear transformation matrix
    M = np.array(((1, k, 0), (0, 1, 0), (0, 0, 1)))
    # take inverse -- useful in inverse-mapping coordinates
    M = np.linalg.inv(M)
    for i in range(img2.shape[0]):
        for j in range(img2.shape[1]):
            # find inverse-mapped coordinates, currently in new origin
            arr = np.dot(M, np.array((j - c_xn, c_y - i, 1)).T)
            # convert from new origin to previous origin (because need intensity values from previous/input image)
            temp = arr[0]
            arr[0] = c_y - arr[1]
            arr[1] = c_x + temp
            # put only if lie in this range, the new image should be black in other places, to account for the transformation
            if arr[0] >= 0 and arr[0] < img.shape[0] and arr[1] >= 0 and arr[1] < img.shape[1]:
                img2[i][j] = img[int(arr[0])][int(arr[1])]
    # return new (transformed) image
    return img2

def part_a(name, ext, theta, axis = 1):
    # read input image file -- in np.array format currently
    img = readImage(name + '.' + ext)
    # apply transformation
    img = rotate(img, theta)
    # get PIL image from np.array
    img = getImage(img)
    # save image will given name
    img.save(name + '_rot.' + ext)
    # get np.array format from PIL image -- useful in applying required transformations
    img = getArray(img)
    # apply transformation
    img = flip(img, axis)
    # get PIL image from np.array
    img = getImage(img)
    # save image will given name
    img.save(name + '_rot_flip.' + ext)

def part_b(name, ext, t_x, t_y):
    # read input image file -- in np.array format currently
    img = readImage(name + '.' + ext)
    # apply transformation
    img = translate(img, t_x, t_y)
    # get PIL image from np.array
    img = getImage(img)
    # save image will given name
    img.save(name + '_translate.' + ext)

def part_c(name, ext, s_x, s_y):
    # read input image file -- in np.array format currently
    img = readImage(name + '.' + ext)
    # apply transformation
    img = scale(img, s_x, s_y)
    # get PIL image from np.array
    img = getImage(img)
    # save image will given name
    img.save(name + '_scale.' + ext)

def part_d(name, ext, k):
    # read input image file -- in np.array format currently
    img = readImage(name + '.' + ext)
    # apply transformation
    img = shear(img, k)
    # get PIL image from np.array
    img = getImage(img)
    # save image will given name
    img.save(name + '_shear.' + ext)
    # get PIL image from np.array
    img = getArray(img)
    # apply transformation
    img = rotate(img, 90)
    # get PIL image from np.array
    img = getImage(img)
    # save image will given name
    img.save(name + '_shear_rot.' + ext)
    img = getArray(img)
    # apply transformation
    img = scale(img, 2, 2)
    # get PIL image from np.array
    img = getImage(img)
    # save image will given name
    img.save(name + '_shear_rot_scale.' + ext)

# input image
try:
    # take input file name from command line
    img = str(sys.argv[1])
except:
    print("Input file not given. Run python3 \"part2.py <imageName>\".")
    exit(1)
# obtaining file name and file extension. Helful in generalising our code.
# Won't have to change anything in the code to run on any image.
# l[0] contains image name and l[1] contains image extension
l = img.split('.')
# Run part 2.a of assignment. 45 is the angle of rotation
part_a(l[0], l[1], 45)
# Run part 2.b of assignment. 32's are translations in x and y axes
part_b(l[0], l[1], 32, 32)
# Run part 2.c of assignment. 3's are scales in x and y axes
part_c(l[0], l[1], 3, 3)
# Run part 2.d of assignment. 0.2 is shear factor in x direction
part_d(l[0], l[1], 0.2)
