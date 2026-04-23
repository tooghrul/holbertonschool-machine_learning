#!/usr/bin/env python3
"""
Module to perform a random crop on an image using TensorFlow.
"""
import tensorflow as tf


def crop_image(image, size):
    """
    Performs a random crop of an image.

    Args:
        image: a 3D tf.Tensor containing the image to crop.
        size: a tuple containing the size (h, w, c) of the crop.

    Returns:
        The randomly cropped image as a tf.Tensor.
    """
    return tf.image.random_crop(image, size=size)
