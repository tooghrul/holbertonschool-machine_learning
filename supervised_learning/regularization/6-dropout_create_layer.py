#!/usr/bin/env python3
"""Module for creating a layer with dropout in TensorFlow."""

import tensorflow as tf


def dropout_create_layer(prev, n, activation, keep_prob, training=True):
    """
    Create a layer of a neural network using dropout.

    Args:
        prev: A tensor containing the output of the previous layer
        n: The number of nodes the new layer should contain
        activation: The activation function for the new layer
        keep_prob: The probability that a node will be kept
        training: A boolean indicating whether the model is in training mode

    Returns:
        The output of the new layer
    """
    # Create kernel initializer as specified in requirements
    initializer = tf.keras.initializers.VarianceScaling(
        scale=2.0,
        mode='fan_avg'
    )

    # Create Dense layer
    dense = tf.keras.layers.Dense(
        units=n,
        activation=activation,
        kernel_initializer=initializer
    )

    # Apply dense layer
    output = dense(prev)

    # Apply dropout
    # dropout_rate = 1 - keep_prob
    dropout = tf.keras.layers.Dropout(rate=1 - keep_prob)
    output = dropout(output, training=training)

    return output
