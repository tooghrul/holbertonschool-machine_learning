#!/usr/bin/env python3
"""Module for performing convolution on grayscale images."""

import numpy as np


def convolve_grayscale(images, kernel, padding='same',
                       stride=(1, 1)):
    """
    Performs a convolution on grayscale images.

    Args:
        images (numpy.ndarray): Array of shape (m, h, w)
        kernel (numpy.ndarray): Array of shape (kh, kw)
        padding (str or tuple): 'same', 'valid', or (ph, pw)
        stride (tuple): (sh, sw)

    Returns:
        numpy.ndarray: Convolved images
    """
    m, h, w = images.shape
    kh, kw = kernel.shape
    sh, sw = stride

    # VALID CASE (special requirement: embed into original size)
    if padding == 'valid':
        out_h = (h - kh) // sh + 1
        out_w = (w - kw) // sw + 1

        temp = np.zeros((m, out_h, out_w))

        for i in range(out_h):
            for j in range(out_w):
                h_start = i * sh
                w_start = j * sw

                region = images[
                    :, h_start:h_start + kh,
                    w_start:w_start + kw
                ]

                temp[:, i, j] = np.sum(
                    region * kernel,
                    axis=(1, 2)
                )

        output = np.zeros((m, h, w))

        h_offset = (h - out_h) // 2
        w_offset = (w - out_w) // 2

        output[
            :, h_offset:h_offset + out_h,
            w_offset:w_offset + out_w
        ] = temp

        return output

    # SAME OR CUSTOM PADDING
    if isinstance(padding, tuple):
        ph, pw = padding

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
        ((0, 0), (ph, ph), (pw, pw)),
        mode='constant'
    )

    out_h = (h + 2 * ph - kh) // sh + 1
    out_w = (w + 2 * pw - kw) // sw + 1

    output = np.zeros((m, out_h, out_w))

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
