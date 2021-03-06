import cv2
import glob

'''NOTE: Please install opencv-contrib-python in your env, works on linux, NOT for windows'''


def detect_and_compute(image):
    descriptor = cv2.xfeatures2d.SIFT_create()
    (kps, features) = descriptor.detectAndCompute(image, None)
    return kps, features


marker = glob.glob('./images/marker.jpg')
marker = cv2.imread(marker[0])
kp1, features_1 = detect_and_compute(marker)
flann = cv2.BFMatcher()
# FLANN_INDEX_KDTREE = 0
# index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
# search_params = dict(checks=50)  # or pass empty dictionary
# linux only comment this line on windows and comment out the next line
# flann = cv2.FlannBasedMatcher(index_params, search_params)

cap = cv2.VideoCapture(0)
while 1:
    _, frame_img = cap.read()
    gray = cv2.cvtColor(frame_img, cv2.COLOR_BGR2GRAY)
    kp2, features_2 = detect_and_compute(gray)

    matches = flann.knnMatch(features_1, features_2, k=2)
    # matches = flann.match(features_1, features_2)
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

cap.release()
cv2.destroyAllWindows()
