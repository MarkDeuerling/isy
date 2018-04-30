import numpy as np
import cv2
from pathlib import Path


def im2double(im):
    """
    Converts uint image (0-255) to double image (0.0-1.0) and generalizes
    this concept to any range.

    :param im:
    :return: normalized image
    """
    min_val = np.min(im.ravel())
    max_val = np.max(im.ravel())
    out = (im.astype('float') - min_val) / (max_val - min_val)
    return out


def make_gaussian(size, fwhm=3, center=None):
    """ Make a square gaussian kernel.

    size is the length of a side of the square
    fwhm is full-width-half-maximum, which
    can be thought of as an effective radius.
    """

    x = np.arange(0, size, 1, float)
    y = x[:,np.newaxis]

    if center is None:
        x0 = y0 = size // 2
    else:
        x0 = center[0]
        y0 = center[1]

    k = np.exp(-4*np.log(2) * ((x-x0)**2 + (y-y0)**2) / fwhm**2)
    return k / np.sum(k)


def convolution_2d(img, kernel):
    """
    Computes the convolution between kernel and image

    :param img: grayscale image
    :param kernel: convolution matrix - 3x3, or 5x5 matrix
    :return: result of the convolution
    """
    # Hint: you need the kernelsize

    offset = int(kernel.shape[0]/2)
    output_img = np.zeros(img.shape)

    image = cv2.copyMakeBorder(img, offset, offset, offset, offset, cv2.BORDER_REPLICATE)
    rows, cols = img.shape[:2]
    for y in np.arange(offset, rows + offset):
        for x in np.arange(offset, cols + offset):
            # extract the ROI of the image by extracting the
            # *center* region of the current (x, y)-coordinates
            # dimensions
            roi = image[y - offset:y + offset + 1, x - offset:x + offset + 1]

            # perform the actual convolution by taking the
            # element-wise multiplicate between the ROI and
            # the kernel, then summing the matrix
            k = (roi * kernel).sum()

            # store the convolved value in the output (x,y)-
            # coordinate of the output image
            output_img[y - offset, x - offset] = k

    return output_img


if __name__ == "__main__":

    # 1. load image in grayscale
    REAL_IMG_PATH = 'images/Lenna.png'
    p = Path(__file__)
    IMG_PATH = str(p.absolute().parent.joinpath(REAL_IMG_PATH))
    # 2. convert image to 0-1 image (see im2double)
    gray_img = cv2.imread(IMG_PATH, 0)
    img_01 = im2double(gray_img)

    # image kernels
    sobelmask_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobelmask_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    gk = make_gaussian(11)

    blur_img = convolution_2d(img_01, gk)

    # 3 .use image kernels on normalized image
    sobel_x = convolution_2d(blur_img, sobelmask_x)
    sobel_y = convolution_2d(blur_img, sobelmask_y)

    # 4. compute magnitude of gradients
    mag = np.sqrt(sobel_x ** 2 + sobel_y ** 2).astype(np.float)
    normalized_img = np.zeros(mag.shape)
    mog = cv2.normalize(mag, normalized_img, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    # Show resulting images
    cv2.imshow("sobel_x", sobel_x)
    cv2.imshow("sobel_y", sobel_y)
    cv2.imshow("mog", mog)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
