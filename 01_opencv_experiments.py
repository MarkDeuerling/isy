import cv2

cap = cv2.VideoCapture(0)
mode = 0


def gaus_thresh(img):
    gray_img_1_channel = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # remove noise
    gray_img_1_channel = cv2.medianBlur(gray_img_1_channel, 5)
    gaus_thresh = cv2.adaptiveThreshold(
        gray_img_1_channel,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11,
        2)
    return gaus_thresh


def otsu_thresh(img):
    # needs to be 1 channel
    gray_img_1_channel = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, otsu_thresh = cv2.threshold(
        gray_img_1_channel, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return otsu_thresh


def canny_edge_detection(img):
    gray_img_1_channel = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray_img_1_channel, 100, 200)
    return edges


exp_map = {
    0: lambda img: img,
    1: lambda img: cv2.cvtColor(img, cv2.COLOR_RGB2HSV),
    2: lambda img: cv2.cvtColor(img, cv2.COLOR_RGB2LAB),
    3: lambda img: cv2.cvtColor(img, cv2.COLOR_RGB2YUV),
    4: lambda img: gaus_thresh(img),
    5: lambda img: otsu_thresh(img),
    6: lambda img: canny_edge_detection(img)
}

while True:
    _, frame_img = cap.read()
    ch = cv2.waitKey(1) & 0xFF

    # origin
    if ch == ord('0'):
        mode = 0

    # hsv
    if ch == ord('1'):
        mode = 1

    # lab
    if ch == ord('2'):
        mode = 2

    # yuv
    if ch == ord('3'):
        mode = 3

    # Gaussian-Thresholding
    if ch == ord('4'):
        mode = 4

    # Otsu-Thresholding
    if ch == ord('5'):
        mode = 5

    # canny edge detection
    if ch == ord('6'):
        mode = 6

    # quit
    if ch == ord('q'):
        break

    frame_img = exp_map[mode](frame_img)
    cv2.imshow('image', frame_img)

cap.release()
cv2.destroyAllWindows()
