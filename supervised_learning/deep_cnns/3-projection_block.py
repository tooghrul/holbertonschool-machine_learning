#!/usr/bin/env python3
"""
Module to create a projection block for a ResNet
"""
from tensorflow import keras as K


def projection_block(A_prev, filters, s=2):
    """
    Builds a projection block as described in Deep Residual Learning (2015)

    Args:
        A_prev: output from the previous layer
        filters: tuple/list containing F11, F3, F12
        s: stride of the first convolution in main and shortcut paths

    Returns:
        The activated output of the projection block
    """
    # Retrieve filters
    F11, F3, F12 = filters

    # Initialize He Normal with seed 0
    init = K.initializers.HeNormal(seed=0)

    # --- Main Path ---
    # First Component: 1x1 convolution with stride s
    X = K.layers.Conv2D(
        filters=F11,
        kernel_size=(1, 1),
        strides=(s, s),
        padding='same',
        kernel_initializer=init
    )(A_prev)
    X = K.layers.BatchNormalization(axis=3)(X)
    X = K.layers.Activation('relu')(X)

    # Second Component: 3x3 convolution
    X = K.layers.Conv2D(
        filters=F3,
        kernel_size=(3, 3),
        padding='same',
        kernel_initializer=init
    )(X)
    X = K.layers.BatchNormalization(axis=3)(X)
    X = K.layers.Activation('relu')(X)

    # Third Component: 1x1 convolution
    X = K.layers.Conv2D(
        filters=F12,
        kernel_size=(1, 1),
        padding='same',
        kernel_initializer=init
    )(X)
    X = K.layers.BatchNormalization(axis=3)(X)

    # --- Shortcut Path ---
    # 1x1 convolution to match dimensions, followed by Batch Normalization
    X_shortcut = K.layers.Conv2D(
        filters=F12,
        kernel_size=(1, 1),
        strides=(s, s),
        padding='same',
        kernel_initializer=init
    )(A_prev)
    X_shortcut = K.layers.BatchNormalization(axis=3)(X_shortcut)

    # --- Final Step ---
    # Add Shortcut to main path and activate
    X = K.layers.Add()([X, X_shortcut])
    X = K.layers.Activation('relu')(X)

    return X
