import cv2
import numpy as np
import glob


# global constants
min_matches = 10
marker = glob.glob('./images/marker.jpg')
marker = cv2.imread(marker[0])

# initialize flann and SIFT extractor
# note unfortunately in the latest OpenCV + python is a minor bug in the flann
# flann = cv2.FlannBasedMatcher(indexParams, {})
# so we use the alternative but slower Brute-Force Matcher BFMatcher


def detect_and_compute(image):
    descriptor = cv2.xfeatures2d.SIFT_create()
    (kps, features) = descriptor.detectAndCompute(image, None)
    return kps, features


flann = cv2.BFMatcher()

# extract marker descriptors
kp_marker, features_marker = detect_and_compute(marker)


def render_virtual_object(img, x0, y0, x1, y1, quad):
    # define vertices, edges and colors of your 3D object, e.g. cube

    vertices = np.float32([[0, 0, 0],
                          [1, 0, 0],
                          [1, 1, 0],
                          [0, 1, 0],

                          [0, 0, -0.5],
                          [1, 0, -0.5],
                          [1, 1, -0.5],
                          [0, 1, -0.5]])
    edges = [(0, 1),
             (1, 2),
             (2, 3),
             (3, 0),

             (0, 4),
             (1, 5),
             (2, 6),
             (3, 7),

             (4, 5),
             (5, 6),
             (6, 7),
             (7, 4)]

    color_lines = (0, 0, 0)

    # define quad plane in 3D coordinates with z = 0
    quad_3d = np.float32([[x0, y0, 0], [x1, y0, 0], [x1, y1, 0], [x0, y1, 0]])

    h, w = img.shape[:2]
    # define intrinsic camera parameter
    K = np.float64([[w, 0, 0.5*(w-1)],
                    [0, w, 0.5*(h-1)],
                    [0, 0, 1.0]])

    # find object pose from 3D-2D point correspondences of the 3d quad using Levenberg-Marquardt optimization
    # in order to work we need K (given above and YOUR distortion coefficients from Assignment 2 (camera calibration))
    # YOUR VALUES HERE
    # dist_coef = np.array([])
    dist_coef = np.zeros((4, 1))

    # compute extrinsic camera parameters using cv2.solvePnP
    _, rot_vec, trans_vec = cv2.solvePnP(quad_3d, quad, K, dist_coef, flags=cv2.SOLVEPNP_ITERATIVE)

    # transform vertices: scale and translate form 0 - 1, in window size of the marker
    scale = [x1 - x0, y1 - y0, x1 - x0]
    trans = [x0, y0, -x1 - x0]

    verts = scale * vertices + trans

    # call cv2.projectPoints with verts, and solvePnP result, K, and dist_coeff
    # returns a tuple that includes the transformed vertices as a first argument
    verts, _ = cv2.projectPoints(verts, rot_vec, trans_vec, K, dist_coef)

    # we need to reshape the result of projectPoints
    verts = verts.reshape(-1, 2)

    # render edges
    for edge in edges:
        x0, y0 = verts[edge[0]]
        x1, y1 = verts[edge[1]]
        cv2.line(img, (int(x0), int(y0)), (int(x1), int(y1)), color_lines, 2)
    

cap = cv2.VideoCapture(0)
cv2.namedWindow('Interactive Systems: AR Tracking')
while 1:
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    _, frame_img = cap.read()
    gray = cv2.cvtColor(frame_img, cv2.COLOR_BGR2GRAY)
    kp_frame, features_frame = detect_and_compute(gray)

    # detect and compute descriptor in camera image
    # and match with marker descriptor
    matches = flann.knnMatch(features_frame, features_marker, k=2)

    # filter matches by distance [Lowe2004]
    matches = [match[0] for match in matches if len(match) == 2 and
               match[0].distance < match[1].distance * 0.75]

    # if there are less than min_matches we just keep going looking
    # early break
    if len(matches) < min_matches:
        cv2.imshow('Interactive Systems: AR Tracking', gray)
        continue
    keypointsMarker = kp_marker
    keypointsFrame = kp_frame
    # extract 2d points from matches data structure
    p0 = [keypointsMarker[m.trainIdx].pt for m in matches]
    p1 = [keypointsFrame[m.queryIdx].pt for m in matches]
    # transpose vectors
    p0, p1 = np.array([p0, p1])

    # we need at least 4 match points to find a homography matrix
    if len(p0) < 4:
        cv2.imshow('Interactive Systems: AR Tracking', frame_img)
        continue

    # find homography using p0 and p1, returning H and status
    # H - homography matrix
    # status - status about inliers and outliers for the plane mapping
    H, mask = cv2.findHomography(p0, p1, cv2.RANSAC, 4.0)

    # on the basis of the status object we can now filter RANSAC outliers
    if mask is None:
        continue
    mask = mask.ravel() != 0
    if mask.sum() < min_matches:
        cv2.imshow('Interactive Systems: AR Tracking', frame_img)
        continue

    # take only inliers - mask of Outlier/Inlier
    p0, p1 = p0[mask], p1[mask]
    # get the size of the marker and form a quad in pixel coords np float array using w/h as the corner points
    w1, h1 = marker_size = marker.shape[:2]
    quad = [[0, 0], [0, h1], [w1, h1], [w1, 0]]

    # perspectiveTransform needs a 3-dimensional array
    quad = np.array([quad], dtype=np.float32)
    quad_transformed = cv2.perspectiveTransform(quad, H)
    # transform back to 2D array
    quad = quad_transformed[0]
    quad_int = quad.astype(dtype=np.int)

    # render quad in image plane and feature points as circle using cv2.polylines + cv2.circle
    cv2.polylines(frame_img, [quad_int], isClosed=True, color=(0, 0, 255), thickness=2)
    for p in p1:
        p = p.astype(np.int)
        cv2.circle(frame_img, (p[0], p[1]), 20, (115, 0, 0))

    # render virtual object on top of quad
    render_virtual_object(frame_img, 0, 0, h1, w1, quad)

    cv2.imshow('Interactive Systems: AR Tracking', frame_img)
