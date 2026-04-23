#!/usr/bin/env python3
"""Module for convolutional forward propagation."""

import numpy as np


def conv_forward(A_prev, W, b, activation,
                 padding="same", stride=(1, 1)):
    """
    Performs forward propagation over a convolutional layer.

    Args:
        A_prev (numpy.ndarray): Shape (m, h_prev, w_prev, c_prev)
        W (numpy.ndarray): Shape (kh, kw, c_prev, c_new)
        b (numpy.ndarray): Shape (1, 1, 1, c_new)
        activation (function): Activation function
        padding (str): 'same' or 'valid'
        stride (tuple): (sh, sw)

    Returns:
        numpy.ndarray: Output of convolutional layer
    """
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw, _, c_new = W.shape
    sh, sw = stride

    # Determine padding
    if padding == 'valid':
        ph, pw = 0, 0

    elif padding == 'same':
        out_h = int(np.ceil(h_prev / sh))
        out_w = int(np.ceil(w_prev / sw))

        ph = int(
            np.ceil(((out_h - 1) * sh + kh - h_prev) / 2)
        )
        pw = int(
            np.ceil(((out_w - 1) * sw + kw - w_prev) / 2)
        )

    else:
        raise ValueError("padding must be 'same' or 'valid'")

    # Pad input
    padded = np.pad(
        A_prev,
        ((0, 0), (ph, ph), (pw, pw), (0, 0)),
        mode='constant'
    )

    # Output dimensions
    out_h = (h_prev + 2 * ph - kh) // sh + 1
    out_w = (w_prev + 2 * pw - kw) // sw + 1

    Z = np.zeros((m, out_h, out_w, c_new))

    # Convolution (3 loops: i, j, filter)
    for i in range(out_h):
        for j in range(out_w):
            h_start = i * sh
            w_start = j * sw

            region = padded[
                :, h_start:h_start + kh,
                w_start:w_start + kw,
                :
            ]

            for k in range(c_new):
                Z[:, i, j, k] = np.sum(
                    region * W[:, :, :, k],
                    axis=(1, 2, 3)
                ) + b[:, :, :, k]

    return activation(Z)
