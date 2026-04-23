#!/usr/bin/env python3
"""Module for performing convolution on multi-channel images."""

import numpy as np


def convolve(images, kernels, padding='same',
             stride=(1, 1)):
    """
    Performs a convolution on images using multiple kernels.

    Args:
        images (numpy.ndarray): Shape (m, h, w, c)
        kernels (numpy.ndarray): Shape (kh, kw, c, nc)
        padding (str or tuple): 'same', 'valid', or (ph, pw)
        stride (tuple): (sh, sw)

    Returns:
        numpy.ndarray: Convolved images of shape
            (m, out_h, out_w, nc)
    """
    m, h, w, c = images.shape
    kh, kw, _, nc = kernels.shape
    sh, sw = stride

    # Handle padding
    if isinstance(padding, tuple):
        ph, pw = padding

    elif padding == 'valid':
        ph, pw = 0, 0

    elif padding == 'same':
        out_h = int(np.ceil(h / sh))
        out_w = int(np.ceil(w / sw))

        ph = int(
            np.ceil(((out_h - 1) * sh + kh - h) / 2)
        )
        pw = int(
            np.ceil(((out_w - 1) * sw + kw - w) / 2)
        )

    else:
        raise ValueError("padding must be 'same', 'valid', "
                         "or a tuple")

    padded = np.pad(
        images,
        ((0, 0), (ph, ph), (pw, pw), (0, 0)),
        mode='constant'
    )

    out_h = (h + 2 * ph - kh) // sh + 1
    out_w = (w + 2 * pw - kw) // sw + 1

    output = np.zeros((m, out_h, out_w, nc))

    # 3-loop constraint:
    # i, j for spatial positions only
    # k for kernel index
    for i in range(out_h):
        for j in range(out_w):
            h_start = i * sh
            w_start = j * sw

            region = padded[
                :, h_start:h_start + kh,
                w_start:w_start + kw,
                :
            ]

            for k in range(nc):
                output[:, i, j, k] = np.sum(
                    region * kernels[:, :, :, k],
                    axis=(1, 2, 3)
                )

    return output
