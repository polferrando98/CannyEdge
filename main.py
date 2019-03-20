import cv2
import numpy as np


def convolve(dest, src, i, j, kernel):
    krows, kcols = kernel.shape
    srctmp = src[i:i + krows, j:j + kcols]
    dest[i, j] = (srctmp * kernel[:, :, np.newaxis]).sum(axis=(0, 1))


def execute():
    # Load an image
    img = cv2.imread("sonic.jpg", cv2.IMREAD_ANYCOLOR)
    rows, cols, channels = img.shape

    # Kernel size / radius
    ksize = 100
    kradi = ksize // 2

    # Create the kernel manually
    # kernel = np.array([
    #     [1.,  4.,  7.,  4., 1.],
    #     [4., 16., 26., 16., 4.],
    #     [7., 26., 41., 26., 7.],
    #     [4., 16., 26., 16., 4.],
    #     [1.,  4.,  7.,  4., 1.]
    # ])

    # Creating the kernel with opencv
    kradi = ksize // 2
    sigma = np.float64(kradi) / 2
    kernel = cv2.getGaussianKernel(ksize, sigma)
    kernel = np.repeat(kernel, ksize, axis=1)
    kernel = kernel * kernel.transpose()
    kernel = kernel / kernel.sum()

    # Create a copy with black padding
    imgpadding = np.zeros((rows + 2 * kradi, cols + 2 * kradi, channels))
    imgpadding[kradi:-kradi, kradi:-kradi] = img

    # Convolution
    filtered = np.zeros(img.shape)
    for i in range(0, rows):
        for j in range(0, cols):
            convolve(filtered, imgpadding, i, j, kernel)
    filtered /= kernel.sum()

    # Using cv2
    # filtered = cv2.GaussianBlur(img, (5,5), 2.0)

    # Show the image
    cv2.imshow("Original", img)
    cv2.imshow("Filtered", np.uint8(filtered))
    cv2.waitKey(0)


if __name__ == "__main__" : execute()
