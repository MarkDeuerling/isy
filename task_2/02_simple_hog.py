import numpy as np
import cv2
import math
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import glob

###############################################################
#
# Write your own descriptor / Histogram of Oriented Gradients
#
###############################################################


def plot_histogram(hist, bins):
    width = 0.7 * (bins[1] - bins[0])
    center = (bins[:-1] + bins[1:]) / 2
    plt.bar(center, hist, align='center', width=width)
    plt.show()


def compute_simple_hog(imgcolor, keypoints):

    # convert color to gray image and extract feature in gray
    gray = cv2.cvtColor(imgcolor, cv2.COLOR_BGR2GRAY)

    # compute x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)

    # compute magnitude and angle of the gradients
    magnitude = cv2.magnitude(sobelx, sobely)
    phase = cv2.phase(sobelx, sobely, True)

    # go through all keypoints and compute feature vector
    descr = np.zeros((len(keypoints), 8), np.float32)
    count = 0
    for kp in keypoints:
        # print kp.pt, kp.size
        print('kp:{}, size:{}'.format(kp.pt, kp.size))
        x = int(kp.pt[0])
        y = int(kp.pt[1])
        size = int(kp.size) >> 2
        # extract angle in keypoint sub window
        phase_kp = phase[x-size:x+size, y-size:y+size]

        # extract gradient magnitude in keypoint subwindow
        magnitude_kp = magnitude[x-size:x+size, y-size:y+size]

        # create histogram of angle in subwindow BUT only where magnitude of gradients is non zero! Why? Find an
        # answer to that question use np.histogram
        non_zero_angle = phase_kp[magnitude_kp != 0]

        (hist, bins) = np.histogram(non_zero_angle, bins=[0, 1, 2, 3, 4, 5, 6, 7, 8])
        plot_histogram(hist, bins)
        descr[count] = hist

    return descr


keypoints = [cv2.KeyPoint(15, 15, 11)]

# test for all test images
test = cv2.imread('./images/hog_test/circle.jpg')
descriptor = compute_simple_hog(test, keypoints)

