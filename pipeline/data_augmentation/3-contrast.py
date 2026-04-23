#!/usr/bin/env python3
"""
Module to randomly adjust image contrast using TensorFlow.
"""
import tensorflow as tf


def change_contrast(image, lower, upper):
    """
    Randomly adjusts the contrast of an image.

    Args:
        image: a 3D tf.Tensor containing the image to adjust.
        lower: float, lower bound for the random contrast factor.
        upper: float, upper bound for the random contrast factor.

    Returns:
        The contrast-adjusted image as a tf.Tensor.
    """
    return tf.image.random_contrast(image, lower, upper)
