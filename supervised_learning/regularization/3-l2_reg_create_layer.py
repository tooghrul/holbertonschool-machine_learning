#!/usr/bin/env python3
"""Module for creating a layer with L2 regularization in TensorFlow."""

import tensorflow as tf


def l2_reg_create_layer(prev, n, activation, lambtha):
    """
    Create a neural network layer in TensorFlow that includes L2
    regularization.

    Args:
        prev: A tensor containing the output of the previous layer
        n: The number of nodes the new layer should contain
        activation: The activation function that should be used on the layer
        lambtha: The L2 regularization parameter

    Returns:
        The output of the new layer
    """
    # Create kernel initializer as specified in requirements
    initializer = tf.keras.initializers.VarianceScaling(
        scale=2.0,
        mode='fan_avg'
    )

    # Create L2 regularizer
    regularizer = tf.keras.regularizers.L2(lambtha)

    # Create Dense layer with L2 regularization on kernel
    layer = tf.keras.layers.Dense(
        units=n,
        activation=activation,
        kernel_initializer=initializer,
        kernel_regularizer=regularizer
    )

    # Return the output of the layer
    return layer(prev)
