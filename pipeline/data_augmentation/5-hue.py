#!/usr/bin/env python3
"""
Module to adjust the hue of an image using TensorFlow.
"""
import tensorflow as tf


def change_hue(image, delta):
    """
    Changes the hue of an image.

    Args:
        image: a 3D tf.Tensor containing the image to change.
        delta: a float representing the amount the hue should change.

    Returns:
        The altered image as a tf.Tensor.
    """
    return tf.image.adjust_hue(image, delta)
