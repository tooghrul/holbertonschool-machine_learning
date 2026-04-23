#!/usr/bin/env python3
import numpy as np

def convolve_grayscale_valid(images, kernel):
    """
    Performs a valid convolution on grayscale images

    images: numpy.ndarray of shape (m, h, w)
    kernel: numpy.ndarray of shape (kh, kw)

    Returns: numpy.ndarray of shape (m, h - kh + 1, w - kw + 1)
    """

    m, h, w = images.shape
    kh, kw = kernel.shape

    # Output dimensions
    output_h = h - kh + 1
    output_w = w - kw + 1

    # Initialize output
    output = np.zeros((m, output_h, output_w))

    # Only TWO loops: over spatial positions
    for i in range(output_h):
        for j in range(output_w):
            # Slice all images at once (vectorized over m)
            slice_ = images[:, i:i+kh, j:j+kw]

            # Element-wise multiply and sum over kernel dims
            output[:, i, j] = np.sum(slice_ * kernel, axis=(1, 2))

    return output
