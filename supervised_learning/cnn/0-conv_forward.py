#!/usr/bin/env python3
"""
Module containing forward propagation for a convolutional layer.
"""
import numpy as np


def conv_forward(A_prev, W, b, activation, padding="same", stride=(1, 1)):
    """
    Performs forward propagation over a convolutional layer of a neural network.

    Args:
        A_prev: numpy.ndarray of shape (m, h_prev, w_prev, c_prev)
            containing the output of the previous layer.
        W: numpy.ndarray of shape (kh, kw, c_prev, c_new)
            containing the kernels for the convolution.
        b: numpy.ndarray of shape (1, 1, 1, c_new)
            containing the biases applied to the convolution.
        activation: function applied to the convolution output.
        padding: string, either "same" or "valid".
        stride: tuple of (sh, sw) containing the strides for the convolution.

    Returns:
        The output of the convolutional layer.
    """
    # Retrieve dimensions from A_prev and W
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw, c_prev, c_new = W.shape
    sh, sw = stride

    # Determine padding values
    if padding == "same":
        ph = ((h_prev - 1) * sh + kh - h_prev) // 2
        pw = ((w_prev - 1) * sw + kw - w_prev) // 2
    else:  # "valid"
        ph, pw = 0, 0

    # Calculate output dimensions
    nh = (h_prev + 2 * ph - kh) // sh + 1
    nw = (w_prev + 2 * pw - kw) // sw + 1

    # Initialize output volume
    output = np.zeros((m, nh, nw, c_new))

    # Apply padding to height and width axes
    A_prev_pad = np.pad(
        A_prev,
        ((0, 0), (ph, ph), (pw, pw), (0, 0)),
        mode='constant',
        constant_values=0
    )

    # Perform the convolution
    for i in range(nh):
        for j in range(nw):
            # Define vertical and horizontal window boundaries
            v_start, v_end = i * sh, i * sh + kh
            h_start, h_end = j * sw, j * sw + kw

            # Extract slice and compute dot product across kernels
            # Axes ([1,2,3], [0,1,2]) maps (kh, kw, c_prev) between slice and W
            a_slice = A_prev_pad[:, v_start:v_end, h_start:h_end, :]
            output[:, i, j, :] = np.tensordot(
                a_slice, W, axes=([1, 2, 3], [0, 1, 2])
            ) + b

    return activation(output)
