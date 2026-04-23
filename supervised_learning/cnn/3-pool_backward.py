#!/usr/bin/env python3
"""
Module containing backpropagation for a pooling layer.
"""
import numpy as np


def pool_backward(dA, A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """
    Performs backpropagation over a pooling layer of a neural network.

    Args:
        dA: numpy.ndarray of shape (m, h_new, w_new, c) containing the
            partial derivatives with respect to the output of the pooling.
        A_prev: numpy.ndarray of shape (m, h_prev, w_prev, c) containing
            the output of the previous layer.
        kernel_shape: tuple of (kh, kw) containing the size of the kernel.
        stride: tuple of (sh, sw) containing the strides for the pooling.
        mode: string, either 'max' or 'avg'.

    Returns:
        dA_prev: partial derivatives with respect to the previous layer.
    """
    m, h_new, w_new, c = dA.shape
    kh, kw = kernel_shape
    sh, sw = stride

    # Initialize dA_prev with zeros
    dA_prev = np.zeros_like(A_prev)

    for i in range(m):  # Loop over examples
        for h in range(h_new):  # Loop over output height
            for w in range(w_new):  # Loop over output width
                for f in range(c):  # Loop over channels
                    # Define the window boundaries
                    v_start, v_end = h * sh, h * sh + kh
                    h_start, h_end = w * sw, w * sw + kw

                    if mode == 'max':
                        # Extract the slice from the forward pass input
                        a_prev_slice = A_prev[i, v_start:v_end,
                                              h_start:h_end, f]
                        # Create a mask for the maximum value
                        mask = (a_prev_slice == np.max(a_prev_slice))
                        # Pass the gradient only to the max position
                        dA_prev[i, v_start:v_end, h_start:h_end, f] += (
                            mask * dA[i, h, w, f]
                        )

                    elif mode == 'avg':
                        # Distribute the gradient equally across the window
                        avg_gradient = dA[i, h, w, f] / (kh * kw)
                        dA_prev[i, v_start:v_end, h_start:h_end, f] += (
                            np.ones((kh, kw)) * avg_gradient
                        )

    return dA_prev
