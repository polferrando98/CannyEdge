import cv2
import numpy as np


def convolve(dest, src, i, j, kernel):
    krows, kcols = kernel.shape
    srctmp = src[i:i + krows, j:j + kcols]
    dest[i, j] = (srctmp * kernel[:, :, np.newaxis]).sum(axis=(0, 1))


def execute():
    # Load an image
    img = cv2.imread("sonic.jpg", cv2.IMREAD_ANYCOLOR)
    rows, cols, _ = img.shape

    # Create the kernel manually
    kernel = np.ones((5, 5))

    # Create a copy with black padding
    imgpadding = np.zeros((rows + 4, cols + 4, 3))
    imgpadding[2:-2, 2:-2] = img

    # # Convolution
    filtered = np.zeros(img.shape)
    for i in range(0, rows):
        for j in range(0, cols):
            convolve(filtered, imgpadding, i, j, kernel)
    filtered /= kernel.sum()

    # Using cv2
    # filtered = cv2.boxFilter(img, -1, (5, 5))

    # Show the image
    cv2.imshow("Original", img)
    cv2.imshow("Filtered", np.uint8(filtered))
    cv2.waitKey(0)


if __name__ == "__main__" : execute()
