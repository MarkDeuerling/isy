import numpy as np
import cv2
import math
import time

from pathlib import Path



######################################################################
# IMPORTANT: Please make yourself comfortable with numpy and python:
# e.g. https://www.stavros.io/tutorials/python/
# https://docs.scipy.org/doc/numpy-dev/user/quickstart.html

# Note: data types are important for numpy and opencv
# most of the time we'll use np.float32 as arrays
# e.g. np.float32([0.1,0.1]) equal np.array([1, 2, 3], dtype='f')


######################################################################
# A2. OpenCV and Transformation and Computer Vision Basic

# (1) read in the image Lenna.png using opencv in gray scale and in color
# and display it NEXT to each other (see result image)
# Note here: the final image displayed must have 3 color channels
#            So you need to copy the gray image values in the color channels
#            of a new image. You can get the size (shape) of an image with rows, cols = img.shape[:2]

# why Lenna? https://de.wikipedia.org/wiki/Lena_(Testbild)

# (2) Now shift both images by half (translation in x) it rotate the colored image by 30 degrees using OpenCV transformation functions
# + do one of the operations on keypress (t - translate, r - rotate, 'q' - quit using cv::warpAffine
# http://docs.opencv.org/3.1.0/da/d54/group__imgproc__transform.html#ga0203d9ee5fcd28d40dbc4a1ea4451983
# Tip: you need to define a transformation Matrix M
# see result image


REAL_IMG_PATH = 'images/Lenna.png'
p = Path(__file__)
IMG_PATH = str(p.absolute().parent.joinpath(REAL_IMG_PATH))

mix_img = cv2.imread(IMG_PATH)
gray_img_1_channel = cv2.cvtColor(mix_img, cv2.COLOR_BGR2GRAY)
gray_img = cv2.cvtColor(gray_img_1_channel, cv2.COLOR_GRAY2RGB)

mix_img = np.concatenate((gray_img, mix_img), axis=1)
rows, cols = mix_img.shape[:2]

while True:
    ch = cv2.waitKey(1) & 0xFF

    # translate
    if ch == ord('t'):
        M = np.float32([[1, 0, 10], [0, 1, 10]])
        mix_img = cv2.warpAffine(mix_img, M, (cols, rows))

    # rotate
    if ch == ord('r'):
        M = cv2.getRotationMatrix2D((cols >> 2, rows >> 2), 20, 1)
        mix_img = cv2.warpAffine(mix_img, M, (cols, rows))

    # quit
    if ch == ord('q'):
        break

    cv2.imshow('image', mix_img)

cv2.destroyAllWindows()
