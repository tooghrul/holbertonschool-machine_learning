#!/usr/bin/env python3
"""
Module containing backpropagation for a convolutional layer.
"""
import numpy as np


def conv_backward(dZ, A_prev, W, b, padding="same", stride=(1, 1)):
    """
    Performs backpropagation over a convolutional layer of a neural network.

    Args:
        dZ: numpy.ndarray of shape (m, h_new, w_new, c_new) containing
            the partial derivatives with respect to the unactivated output.
        A_prev: numpy.ndarray of shape (m, h_prev, w_prev, c_prev).
        W: numpy.ndarray of shape (kh, kw, c_prev, c_new).
        b: numpy.ndarray of shape (1, 1, 1, c_new).
        padding: string, either "same" or "valid".
        stride: tuple of (sh, sw).

    Returns:
        dA_prev, dW, db: partial derivatives with respect to the previous
        layer, the kernels, and the biases, respectively.
    """
    m, h_new, w_new, c_new = dZ.shape
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw, _, _ = W.shape
    sh, sw = stride

    # Initialize gradients
    dA_prev = np.zeros_like(A_prev)
    dW = np.zeros_like(W)
    db = np.sum(dZ, axis=(0, 1, 2), keepdims=True)

    # Padding logic
    if padding == "same":
        ph = ((h_prev - 1) * sh + kh - h_prev) // 2 + 1
        pw = ((w_prev - 1) * sw + kw - w_prev) // 2 + 1
    else:
        ph, pw = 0, 0

    # Pad A_prev and dA_prev
    A_prev_pad = np.pad(A_prev, ((0, 0), (ph, ph), (pw, pw), (0, 0)),
                        mode='constant')
    dA_prev_pad = np.pad(dA_prev, ((0, 0), (ph, ph), (pw, pw), (0, 0)),
                         mode='constant')

    for i in range(h_new):
        for j in range(w_new):
            # Define window boundaries
            v_start, v_end = i * sh, i * sh + kh
            h_start, h_end = j * sw, j * sw + kw

            # Extract current slice
            a_slice = A_prev_pad[:, v_start:v_end, h_start:h_end, :]

            # Update gradient for weights
            # Sum over examples (m) and output spatial locations
            for k in range(c_new):
                dz_val = dZ[:, i, j, k, np.newaxis, np.newaxis, np.newaxis]
                dW[:, :, :, k] += np.sum(a_slice * dz_val, axis=0)

            # Update gradient for padded input
            for k in range(c_new):
                dz_val = dZ[:, i, j, k, np.newaxis, np.newaxis, np.newaxis]
                dA_prev_pad[:, v_start:v_end, h_start:h_end, :] += (
                    W[:, :, :, k] * dz_val
                )

    # Unpad dA_prev
    if ph > 0 and pw > 0:
        dA_prev = dA_prev_pad[:, ph:-ph, pw:-pw, :]
    elif ph > 0:
        dA_prev = dA_prev_pad[:, ph:-ph, :, :]
    elif pw > 0:
        dA_prev = dA_prev_pad[:, :, pw:-pw, :]
    else:
        dA_prev = dA_prev_pad

    return dA_prev, dW, db
