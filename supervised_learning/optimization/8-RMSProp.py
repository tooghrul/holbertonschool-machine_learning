#!/usr/bin/env python3
"""Module to create an RMSProp optimizer in TensorFlow."""
import tensorflow as tf


def create_RMSProp_op(alpha, beta2, epsilon):
    """
    Creates a TensorFlow optimizer using RMSProp.

    Parameters:
    - alpha: learning rate
    - beta2: RMSProp discounting factor (decay rate)
    - epsilon: small number to avoid division by zero

    Returns:
    - optimizer: a TensorFlow RMSProp optimizer instance
    """
    optimizer = tf.keras.optimizers.RMSprop(
        learning_rate=alpha,
        rho=beta2,
        epsilon=epsilon
    )
    return optimizer
