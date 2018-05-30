import cv2
import glob
import numpy as np

'''NOTE: Please install opencv-contrib-python in your env'''


def detect_and_compute(image):
    descriptor = cv2.xfeatures2d.SIFT_create()
    (kps, features) = descriptor.detectAndCompute(image, None)
    # kps = np.float32([kp.pt for kp in kps])
    return kps, features


marker = glob.glob('./images/marker.jpg')
marker = cv2.imread(marker[0])

'''
NOTE: If you'd like to use the vedoe capture comment out line 24, 26, 27 
and comment line 30

'''

# cap = cv2.VideoCapture(0)
while True:
    # _, frame_img = cap.read()
    # gray = cv2.cvtColor(frame_img, cv2.COLOR_BGR2GRAY)
    sift = cv2.xfeatures2d.SIFT_create()
    key_points = sift.detect(marker, None)
    gray = marker

    kp1, features_1 = detect_and_compute(marker)
    kp2, features_2 = detect_and_compute(gray)

    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)  # or pass empty dictionary
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(features_2, features_2, 2)
    matchesMask = [[0, 0] for i in range(len(matches))]
    for i, (m, n) in enumerate(matches):
        if m.distance < 0.7 * n.distance:
            matchesMask[i] = [1, 0]
    draw_params = dict(matchColor=(0, 255, 0),
                       singlePointColor=(255, 0, 0),
                       matchesMask=matchesMask,
                       flags=0)
    img3 = cv2.drawMatchesKnn(marker, kp1, gray, kp2, matches, None, **draw_params)
    cv2.imshow('image', img3)

    # quit
    ch = cv2.waitKey(1) & 0xFF
    if ch == ord('q'):
        break

# cap.release()
cv2.destroyAllWindows()
