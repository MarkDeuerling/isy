import cv2
import glob
import numpy as np
# from Queue import PriorityQueue

############################################################
#
#              Simple Image Retrieval
#
############################################################


# implement distance function
def distance(a, b):
    return cv2.norm(a, b, cv2.NORM_L2)


def create_keypoints(w, h):
    keypoints = []
    keypointSize = 11

    for x in range(0, w, keypointSize):
        for y in range(0, h, keypointSize):
            kp = cv2.KeyPoint(x, y, keypointSize)
            keypoints.append(kp)

    return keypoints


# 1. preprocessing and load
images = glob.glob('./images/db/*/*.jpg')

# 2. create keypoints on a regular grid (cv2.KeyPoint(r, c, keypointSize), as keypoint size use e.g. 11)
descriptors = []
keypoints = create_keypoints(256, 256)

# 3. use the keypoints for each image and compute SIFT descriptors
#    for each keypoint. this compute one descriptor for each image.
sift = cv2.xfeatures2d.SIFT_create()
desc_img = []
for img in images:
    img = cv2.imread(img)
    descriptor = sift.compute(img, keypoints)
    # descriptors.append(descriptor)
    desc_img.append((descriptor, img))

# 4. use one of the query input image to query the 'image database' that
#    now compress to a single area. Therefore extract the descriptor and
#    compare the descriptor to each image in the database using the L2-norm
#    and save the result into a priority queue (q = PriorityQueue())

query_img = 'query_face'
query_img = glob.glob('./images/db/{}.jpg'.format(query_img))[0]
query_img = cv2.imread(query_img)
query_descriptor = sift.compute(query_img, keypoints)

dists = []
for d_i in desc_img:
    desc = d_i[0]
    d = distance(query_descriptor[1], desc[1])
    dists.append((d, d_i[1]))

# for descriptor in descriptors:
#     d = distance(query_descriptor[1], descriptor[1])
#     dists.append((d, descriptor))
sorted_dists = sorted(dists, key=lambda x: x[0])
# 5. output (save and/or display) the query results in the order of smallest distance


def query():
    for q in sorted_dists:
        yield q


next_img = query()
cur_img = next(next_img)
while True:
    ch = cv2.waitKey(1) & 0xFF
    if ch == ord('q'):
        break
    if ch == ord('n'):
        try:
            cur_img = next(next_img)
        except Exception:
            print("No more Images.")
    cv2.imshow('query img', query_img)
    cv2.imshow('match img', cur_img[1])

cv2.destroyAllWindows()