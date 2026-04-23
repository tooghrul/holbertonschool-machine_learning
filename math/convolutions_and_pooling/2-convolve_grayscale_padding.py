#!/usr/bin/env python3
"""Module for performing padded convolution on grayscale images."""

import numpy as np


def convolve_grayscale_padding(images, kernel, padding):
    """
    Performs a convolution on grayscale images with custom padding.

    Args:
        images (numpy.ndarray): Array of shape (m, h, w) containing
            grayscale images.
        kernel (numpy.ndarray): Array of shape (kh, kw) containing
            the convolution kernel.
        padding (tuple): Tuple (ph, pw)
            ph: padding for height
            pw: padding for width

    Returns:
        numpy.ndarray: Convolved images with padding applied
    """
    m, h, w = images.shape
    kh, kw = kernel.shape
    ph, pw = padding

    padded_images = np.pad(
        images,
        ((0, 0), (ph, ph), (pw, pw)),
        mode='constant'
    )

    output_h = h + 2 * ph - kh + 1
    output_w = w + 2 * pw - kw + 1

    output = np.zeros((m, output_h, output_w))

    for i in range(output_h):
        for j in range(output_w):
            image_slice = padded_images[
                :, i:i + kh, j:j + kw
            ]
            output[:, i, j] = np.sum(
                image_slice * kernel,
                axis=(1, 2)
            )

    return output
