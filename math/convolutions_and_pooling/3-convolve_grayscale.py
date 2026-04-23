#!/usr/bin/env python3
"""Module for grayscale convolution."""

import numpy as np


def convolve_grayscale(images, kernel, padding='same',
                       stride=(1, 1)):
    """
    Performs convolution on grayscale images.

    Args:
        images (numpy.ndarray): (m, h, w)
        kernel (numpy.ndarray): (kh, kw)
        padding (str or tuple): 'same', 'valid', or (ph, pw)
        stride (tuple): (sh, sw)

    Returns:
        numpy.ndarray: Convolved output
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
        out_h = int(np.ceil(h / sh))
        out_w = int(np.ceil(w / sw))

        ph = int(np.ceil(((out_h - 1) * sh + kh - h) / 2))
        pw = int(np.ceil(((out_w - 1) * sw + kw - w) / 2))

    else:
        raise ValueError("Invalid padding")

    # Apply padding
    padded = np.pad(
        images,
        ((0, 0), (ph, ph), (pw, pw)),
        mode='constant'
    )

    # Compute output size
    out_h = (h + 2 * ph - kh) // sh + 1
    out_w = (w + 2 * pw - kw) // sw + 1

    output = np.zeros((m, out_h, out_w))

    # Convolution (2 loops only)
    for i in range(out_h):
        for j in range(out_w):
            h_start = i * sh
            w_start = j * sw

            region = padded[
                :, h_start:h_start + kh,
                w_start:w_start + kw
            ]

            output[:, i, j] = np.sum(
                region * kernel,
                axis=(1, 2)
            )

    return output
