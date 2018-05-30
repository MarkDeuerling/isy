import cv2
import glob
import numpy as np

'''NOTE: Please install opencv-contrib-python in your env'''


def match_keypoints(kpsPano1, kpsPano2, descriptors1, descriptors2):
    """This function computes the matching of image features between two different
    images and a transformation matrix (aka homography) that we will use to unwarp the images
    and place them correctly next to each other. There is no need for modifying this, we will
    cover what is happening here later in the course.
    """
    # compute the raw matches using a Bruteforce matcher that
    # compares image descriptors/feature vectors in high-dimensional space
    # by employing K-Nearest-Neighbor match (more next course)

    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)  # or pass empty dictionary

    flann = cv2.FlannBasedMatcher(index_params, search_params)
    rawmatches = flann.knnMatch(descriptors1, descriptors2, 2)

    # bf = cv2.BFMatcher()
    # rawmatches = bf.knnMatch(descriptors1, descriptors2, 2)
    matches = []

    # loop over the raw matches and filter them
    for m in rawmatches:
        if len(m) == 2 and m[0].distance < m[1].distance * 0.75:
            matches.append((m[0].trainIdx, m[0].queryIdx))

    # we need to compute a homography - more next course
    # computing a homography requires at least 4 matches
    if len(matches) > 4:
        # construct the two sets of points
        ptsPano1 = np.float32([kpsPano1[i].pt for (_, i) in matches])
        ptsPano2 = np.float32([kpsPano2[i].pt for (i, _) in matches])

        # compute the homography between the two sets of points
        H, status = cv2.findHomography(ptsPano1, ptsPano2, cv2.RANSAC,
                                       4.0)

        # we return the corresponding perspective transform and some
        # necessary status object + the used matches
        return H, status, matches

    # otherwise, no homograpy could be computed
    return None


def draw_matches(img1, img2, kp1, kp2, matches, status):
    """For each pair of points we draw a line between both images and circles,
    then connect a line between them.
    Returns a new image containing the visualized matches
    """

    (h1, w1) = img1.shape[:2]
    (h2, w2) = img2.shape[:2]
    vis = np.zeros((max(h1, h2), w1 + w2, 3), dtype="uint8")
    vis[0:h1, 0:w1] = img1
    vis[0:h2, w1:] = img2

    for ((idx2, idx1), s) in zip(matches, status):
        # only process the match if the keypoint was successfully matched
        if s == 1:
            # x - columns
            # y - rows
            (x1, y1) = kp1[idx1].pt
            (x2, y2) = kp2[idx2].pt

            # Draw a small circle at both co-ordinates
            # radius 4
            # colour blue
            # thickness = 1
            cv2.circle(vis, (int(x1), int(y1)), 4, (255, 255, 0), 1)
            cv2.circle(vis, (int(x2) + w1, int(y2)), 4, (255, 255, 0), 1)

            # Draw a line in between the two points
            # thickness = 1
            # colour blue
            cv2.line(vis, (int(x1), int(y1)), (int(x2) + w1, int(y2)),
                     (255, 0, 0), 1)
    return vis


def detect_and_compute(image):
    descriptor = cv2.xfeatures2d.SIFT_create()
    (kps, features) = descriptor.detectAndCompute(image, None)
    # kps = np.float32([kp.pt for kp in kps])
    return kps, features


marker = glob.glob('./images/marker.jpg')
marker = cv2.imread(marker[0])

'''
NOTE: If you'd like to use the vedoe capture comment out line 104, 106, 107 
and comment line 110

'''

# cap = cv2.VideoCapture(0)
while True:
    # _, frame_img = cap.read()
    # gray = cv2.cvtColor(frame_img, cv2.COLOR_BGR2GRAY)
    sift = cv2.xfeatures2d.SIFT_create()
    key_points = sift.detect(marker, None)
    gray = marker  

    kps_1, features_1 = detect_and_compute(marker)
    kps_2, features_2 = detect_and_compute(gray)

    M = match_keypoints(kps_1, kps_2, features_1, features_2)
    if M is None:
        break
    H, status, matches = M

    vis = draw_matches(marker, gray, kps_1, kps_2, matches, status)
    cv2.imshow('image', vis)

    # quit
    ch = cv2.waitKey(1) & 0xFF
    if ch == ord('q'):
        break

# cap.release()
cv2.destroyAllWindows()
