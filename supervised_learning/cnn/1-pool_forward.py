#!/usr/bin/env python3
"""
Module containing forward propagation for a pooling layer.
"""
import numpy as np


def pool_forward(A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """
    Performs forward propagation over a pooling layer of a neural network.

    Args:
        A_prev: numpy.ndarray of shape (m, h_prev, w_prev, c_prev)
            containing the output of the previous layer.
        kernel_shape: tuple of (kh, kw) containing the size of the kernel.
        stride: tuple of (sh, sw) containing the strides for the pooling.
        mode: string, either 'max' or 'avg'.

    Returns:
        The output of the pooling layer.
    """
    # Retrieve dimensions
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw = kernel_shape
    sh, sw = stride

    # Calculate output dimensions
    # Pooling usually uses 'valid' style padding (no padding)
    nh = (h_prev - kh) // sh + 1
    nw = (w_prev - kw) // sw + 1

    # Initialize output volume
    output = np.zeros((m, nh, nw, c_prev))

    # Perform pooling
    for i in range(nh):
        for j in range(nw):
            # Define window boundaries
            v_start, v_end = i * sh, i * sh + kh
            h_start, h_end = j * sw, j * sw + kw

            # Extract slice from input
            a_slice = A_prev[:, v_start:v_end, h_start:h_end, :]

            # Apply pooling operation based on mode
            if mode == 'max':
                output[:, i, j, :] = np.max(a_slice, axis=(1, 2))
            elif mode == 'avg':
                output[:, i, j, :] = np.mean(a_slice, axis=(1, 2))

    return output
