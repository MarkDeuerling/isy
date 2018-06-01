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


def initialize(img, cluster_colors):
    """inittialize the current_cluster_centers array for each cluster with a random pixel position"""
    k = 3
    c = []
    for i in range(k):
        x = np.random.randint(0, np.max(img))
        y = np.random.randint(0, np.max(img))
        c.append(img[x, y])
    # current_cluster_centers = np.array(list(zip(c_x, c_y)), dtype=np.float32)
    # c = dict(enumerate(current_cluster_centers))
    c = np.array(c)
    if cluster_colors:
        print(cluster_colors)
        return cluster_colors[:k]
    print(c)
    return c


def kmeans(img, cluster_colors):
    """Main k-means function iterating over max_iterations and stopping if
    the error rate of change is less then 2% for consecutive iterations, i.e. the
    algorithm converges. In our case the overall error might go up and down a little
    since there is no guarantee we find a global minimum.
    """
    max_iter = 10
    max_change_rate = 0.02
    dist = sys.float_info.max

    c = initialize(img, cluster_colors)
    print(len(c))
    
    def rek_fun(c, img2, iterations=0, max_iteration=3):
        result = np.zeros((h1, w1, 3), np.uint8)
        dist_list = []
        clusters = [[]] * len(c)
        for x in range(w1):
            for y in range(h1):
                pix = img2[x, y]
                for i in range(len(c)):
                    group_pix = c[i]
                    dist = distance(pix, group_pix)
                    dist_list.append((i, dist))
                label = min(dist_list, key=lambda z: z[1])
                dist_list.clear()
                centroid = label[0]
                list_c = clusters[centroid]
                list_c.append(pix)
                clusters[centroid] = list_c[:]
                result[x, y] = c[centroid]

        clusters = np.array(clusters)
        a = []
        for i in range(len(c)):
            mean = np.mean(clusters[i], axis=0)
            a.append(mean)
        iterations += 1
        print(iterations)
        if iterations == max_iteration:
            return a, result
        return rek_fun(a, result, iterations)

    _, img = rek_fun(c, img)
    return img


# num of cluster
numclusters = 3
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
    2: lambda img: cv2.cvtColor(img, cv2.COLOR_RGB2LAB),
    3: lambda img: cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
}
image = exp_map[1](image)
res = kmeans(image, cluster_colors)

h1, w1 = res.shape[:2]
h2, w2 = image.shape[:2]
vis = np.zeros((max(h1, h2), w1 + w2, 3), np.uint8)
vis[:h1, :w1] = res
vis[:h2, w1:w1 + w2] = image

cv2.imshow("Color-based Segmentation Kmeans-Clustering", vis)
cv2.waitKey(0)
cv2.destroyAllWindows()
