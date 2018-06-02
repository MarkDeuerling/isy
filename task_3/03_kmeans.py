import numpy as np
import cv2
import math
import sys
from copy import deepcopy
from matplotlib import pyplot as plt
plt.rcParams['figure.figsize'] = (16, 9)
plt.style.use('ggplot')



############################################################
#
#                       KMEANS
#
############################################################

# implement distance metric - e.g. squared distances between pixels
def distance(a, b):
    return np.linalg.norm(a-b)

# k-means works in 3 steps
# 1. initialize
# 2. assign each data element to current mean (cluster center)
# 3. update mean
# then iterate between 2 and 3 until convergence, i.e. until ~smaller than 5% change rate in the error


def update_mean(img, clustermask):
    """This function should compute the new cluster center, i.e. numcluster mean colors"""
    pass


def assign_to_current_mean(img, result, clustermask):
    """The function expects the img, the resulting image and a clustermask.
    After each call the pixels in result should contain a cluster_color corresponding to the cluster
    it is assigned to. clustermask contains the cluster id (int [0...num_clusters]
    Return: the overall error (distance) for all pixels to there closest cluster center (mindistance px - cluster center).
    """
    overall_dist = 0



    return overall_dist


def initialize(img):
    """inittialize the current_cluster_centers array for each cluster with a random pixel position"""
    k = 64
    c = []
    for i in range(k):
        x = np.random.randint(0, np.max(img))
        y = np.random.randint(0, np.max(img))
        c.append(img[x, y])
    c = np.array(c)

    print(c)
    return c


def kmeans(img, cluster_colors):
    """Main k-means function iterating over max_iterations and stopping if
    the error rate of change is less then 2% for consecutive iterations, i.e. the
    algorithm converges. In our case the overall error might go up and down a little
    since there is no guarantee we find a global minimum.
    """
    max_change_rate = 0.2
    Z = img.reshape((-1, 3))

    c = initialize(img)
    clusters = [[]] * len(c)
    counter = 0
    while 1:
        counter += 1
        if counter == 10:
            break

        clusters = [[]] * len(c)
        for idx, rgb in enumerate(Z):
            dist_list = []
            for i, clust in enumerate(c):
                dist = distance(clust, rgb)
                dist_list.append((i, dist))
            label = min(dist_list, key=lambda z: z[1])
            clusters[label[0]] = clusters[label[0]] + [idx]

        new_cent = []
        for c_i in range(len(c)):
            cent = np.mean([rgb for rgb in Z[clusters[c_i]]], axis=0)
            new_cent.append(cent)
        new_cent = np.array(new_cent)
        error = distance(c, new_cent)
        print('error ', error)
        if error <= max_change_rate:
            break
        c = new_cent

    Z = np.copy(Z)
    if cluster_colors:
        c = cluster_colors[:len(c)]
    for i in range(len(clusters)):
        cluster_rgb = c[i]
        for img_idx in clusters[i]:
            Z[img_idx] = cluster_rgb
    result = Z.reshape(img.shape)
    return result

# num of cluster
numclusters = 6
# corresponding colors for each cluster
cluster_colors = [[255, 0, 0], [0, 255, 0], [0, 0, 255], [0, 255, 255], [255, 255, 255], [0, 0, 0], [128, 128, 128]]
# initialize current cluster centers (i.e. the pixels that represent a cluster center)
current_cluster_centers = np.zeros((numclusters, 1, 3), np.float32)

# load image
imgraw = cv2.imread('./images/Lenna.png')
scaling_factor = 0.5
imgraw = cv2.resize(imgraw, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA)


# compare different color spaces and their result for clustering
# YOUR CODE HERE or keep going with loaded RGB colorspace img = imgraw
image = imgraw
h1, w1 = image.shape[:2]

# execute k-means over the image
# it returns a result image where each pixel is color with one of the cluster_colors
# depending on its cluster assignment
# image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

exp_map = {
    0: lambda img: img,
    1: lambda img: cv2.cvtColor(img, cv2.COLOR_RGB2HSV),
    2: lambda img: cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
}
image = exp_map[0](image)
# 2nd param cluster_colors or None
# res = kmeans(image, cluster_colors)
res = kmeans(image, None)

h1, w1 = res.shape[:2]
h2, w2 = image.shape[:2]
vis = np.zeros((max(h1, h2), w1 + w2, 3), np.uint8)
vis[:h1, :w1] = res
vis[:h2, w1:w1 + w2] = image

cv2.imshow("Color-based Segmentation Kmeans-Clustering", vis)
cv2.waitKey(0)
cv2.destroyAllWindows()
