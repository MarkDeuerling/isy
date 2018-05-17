import numpy as np
import cv2
import math
import sys
from pathlib import Path
from ImageStitcher import *

############################################################
#
#                   Image Stitching
#
############################################################

# 1. load panorama images
REAL_IMG_PATH_1 = 'images/pano1.jpg'
REAL_IMG_PATH_2 = 'images/pano2.jpg'
REAL_IMG_PATH_3 = 'images/pano3.jpg'
p = Path(__file__)
IMG_PATH_1 = str(p.absolute().parent.joinpath(REAL_IMG_PATH_1))
IMG_PATH_2 = str(p.absolute().parent.joinpath(REAL_IMG_PATH_2))
IMG_PATH_3 = str(p.absolute().parent.joinpath(REAL_IMG_PATH_3))
pano_1 = cv2.imread(IMG_PATH_1)
pano_2 = cv2.imread(IMG_PATH_2)
pano_3 = cv2.imread(IMG_PATH_3)

# order of input images is important is important (from right to left)
imageStitcher = ImageStitcher([pano_3, pano_2, pano_1])  # list of images
matchlist, result = imageStitcher.stitch_to_panorama()

if matchlist is None:
    print("We have not enough matching keypoints to create a panorama")
else:
    while True:
        ch = cv2.waitKey(1) & 0xFF
        if ch == ord('q'):
            break
        # cv2.imshow('image', pano_1)
        # cv2.imshow('image2', pano_2)
        # cv2.imshow('matches keypoint', matchlist)
        # cv2.imshow('panorama', result)
        res_key = cv2.resize(matchlist, (800, 400), interpolation=cv2.INTER_CUBIC)
        res_pano = cv2.resize(result, (1200, 400), interpolation=cv2.INTER_CUBIC)
        cv2.imshow('matchelist', res_key)
        cv2.imshow('panorama', res_pano)
    cv2.destroyAllWindows()
