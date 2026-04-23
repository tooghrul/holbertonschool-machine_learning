#!/usr/bin/env python3
"""Module for performing convolution on grayscale images."""

import numpy as np


def convolve_grayscale(images, kernel, padding='same',
                       stride=(1, 1)):
    """
    Performs a convolution on grayscale images.

    Args:
        images (numpy.ndarray): Shape (m, h, w)
        kernel (numpy.ndarray): Shape (kh, kw)
        padding (str or tuple): 'same', 'valid', or (ph, pw)
        stride (tuple): (sh, sw)

    Returns:
        numpy.ndarray: Convolved images
    """
    m, h, w = images.shape
    kh, kw = kernel.shape
    sh, sw = stride

    # Determine padding
    if isinstance(padding, tuple):
        ph, pw = padding

    elif padding == 'valid':
        ph, pw = 0, 0

    elif padding == 'same':
        output_h = int(np.ceil(h / sh))
        output_w = int(np.ceil(w / sw))

        ph = int(
            max((output_h - 1) * sh + kh - h, 0) / 2
        )
        pw = int(
            max((output_w - 1) * sw + kw - w, 0) / 2
        )

    else:
        raise ValueError("padding must be 'same', 'valid', "
                         "or a tuple")

    # Pad images
    padded = np.pad(
        images,
        ((0, 0), (ph, ph), (pw, pw)),
        mode='constant'
    )

    # Compute output dimensions
    output_h = (h + 2 * ph - kh) // sh + 1
    output_w = (w + 2 * pw - kw) // sw + 1

    output = np.zeros((m, output_h, output_w))

    # Two loops only
    for i in range(output_h):
        for j in range(output_w):
            h_start = i * sh
            w_start = j * sw

            image_slice = padded[
                :, h_start:h_start + kh,
                w_start:w_start + kw
            ]

            output[:, i, j] = np.sum(
                image_slice * kernel,
                axis=(1, 2)
            )

    return output
