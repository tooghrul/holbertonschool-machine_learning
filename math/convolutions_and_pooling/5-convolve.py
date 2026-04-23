#!/usr/bin/env python3
"""Module for performing convolution on images using multiple kernels."""

import numpy as np


def convolve(images, kernels, padding='same', stride=(1, 1)):
    """
    Perform a convolution on images with channels using multiple kernels.

    Args:
        images (numpy.ndarray): Array of shape (m, h, w, c) containing
            multiple images, where m is the number of images, h is the
            height in pixels, w is the width in pixels, and c is the
            number of channels.
        kernels (numpy.ndarray): Array of shape (kh, kw, c, nc) containing
            the kernels for the convolution, where kh is the height, kw is
            the width of a kernel, c is the number of channels, and nc is
            the number of kernels.
        padding (tuple, str): Either a tuple of (ph, pw), 'same', or
            'valid'. If 'same', performs a same convolution. If 'valid',
            performs a valid convolution with no padding. If a tuple, ph
            is the padding for the height and pw is the padding for the
            width. The image is padded with 0's. Defaults to 'same'.
        stride (tuple): A tuple of (sh, sw) where sh is the stride for the
            height and sw is the stride for the width. Defaults to (1, 1).

    Returns:
        numpy.ndarray: Array containing the convolved images.

    """
    # Extract dimensions from input images and kernels
    m, h, w, c = images.shape
    kh, kw, c_k, nc = kernels.shape
    sh, sw = stride

    # Determine padding values based on padding parameter
    if padding == 'valid':
        # No padding for valid convolution
        ph, pw = 0, 0
    elif padding == 'same':
        # Same padding: output size equals input size (when stride is 1)
        # Formula: pad = (kernel_size - 1) // 2
        ph = (kh - 1) // 2
        pw = (kw - 1) // 2
    else:
        # Explicit padding values provided as a tuple
        ph, pw = padding

    # Pad the images with zeros along height and width dimensions
    # Shape goes from (m, h, w, c) to (m, h + 2*ph, w + 2*pw, c)
    padded = np.pad(
        images,
        ((0, 0), (ph, ph), (pw, pw), (0, 0)),
        mode='constant'
    )

    # Calculate dimensions after padding
    padded_h = h + 2 * ph
    padded_w = w + 2 * pw

    # Calculate output dimensions using convolution formula:
    # output_size = (input_size - kernel_size) // stride + 1
    output_h = (padded_h - kh) // sh + 1
    output_w = (padded_w - kw) // sw + 1

    # Initialize output array with zeros
    output = np.zeros((m, output_h, output_w, nc))

    # Perform convolution using only two for loops
    # Loops iterate over output spatial dimensions
    for i in range(output_h):
        for j in range(output_w):
            # Extract the receptive field (patch) for all images
            # Shape: (m, kh, kw, c)
            patch_start_h = i * sh
            patch_start_w = j * sw
            patches = padded[
                :,
                patch_start_h:patch_start_h + kh,
                patch_start_w:patch_start_w + kw,
                :
            ]

            # Perform element-wise multiplication between patches and kernels
            # Patches shape (m, kh, kw, c) broadcasts with kernels shape
            # (kh, kw, c, nc) to produce (m, kh, kw, c, nc)
            # Sum over the kernel spatial dimensions and channels
            # (axes 1, 2, and 3) to get result shape: (m, nc)
            output[:, i, j, :] = np.sum(patches * kernels, axis=(1, 2, 3))

    return output
