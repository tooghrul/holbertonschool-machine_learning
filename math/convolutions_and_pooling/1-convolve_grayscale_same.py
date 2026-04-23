#!/usr/bin/env python3
"""
Module for performing same convolution on grayscale images.
"""

import numpy as np


def convolve_grayscale_same(images, kernel):
    """
    Performs a same convolution on grayscale images.

    The output has the same height and width as the input images.
    Zero-padding is applied if necessary.

    Args:
        images (numpy.ndarray): Array of shape (m, h, w)
            - m: number of images
            - h: height of images
            - w: width of images
        kernel (numpy.ndarray): Array of shape (kh, kw)
            - kh: kernel height
            - kw: kernel width

    Returns:
        numpy.ndarray: Convolved images of shape (m, h, w)
    """
    m, h, w = images.shape
    kh, kw = kernel.shape

    # Compute padding for "same" convolution
    pad_h = kh // 2
    pad_w = kw // 2

    # Apply zero padding
    padded = np.pad(
        images,
        pad_width=((0, 0), (pad_h, pad_h), (pad_w, pad_w)),
        mode='constant'
    )

    # Initialize output array
    output = np.zeros((m, h, w))

    # Perform convolution using only two loops (over kernel)
    for i in range(kh):
        for j in range(kw):
            output += kernel[i, j] * padded[
                :,
                i:i + h,
                j:j + w
            ]

    return output
