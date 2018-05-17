import cv2

cap = cv2.VideoCapture(0)
while True:
    _, frame_img = cap.read()
    gray = cv2.cvtColor(frame_img, cv2.COLOR_BGR2GRAY)
    sift = cv2.xfeatures2d.SIFT_create()
    key_points = sift.detect(gray, None)
    # bad python binding
    img = cv2.drawKeypoints(gray, key_points, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    ch = cv2.waitKey(1) & 0xFF
    # quit
    if ch == ord('q'):
        break

    cv2.imshow('image', img)

cap.release()
cv2.destroyAllWindows()
