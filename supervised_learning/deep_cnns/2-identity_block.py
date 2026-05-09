#!/usr/bin/env python3
"""
Module to create an identity block for a ResNet
"""
from tensorflow import keras as K


def identity_block(A_prev, filters):
    """
    Builds an identity block as described in Deep Residual Learning (2015)
    
    Args:
        A_prev: output from the previous layer
        filters: tuple/list containing F11, F3, F12 (filters for conv layers)
        
    Returns:
        The activated output of the identity block
    """
    # Retrieve filters
    F11, F3, F12 = filters
    
    # Initialize He Normal with seed 0
    init = K.initializers.HeNormal(seed=0)
    
    # --- First Component of Main Path ---
    # 1x1 convolution to reduce dimensionality
    X = K.layers.Conv2D(
        filters=F11,
        kernel_size=(1, 1),
        padding='same',
        kernel_initializer=init
    )(A_prev)
    X = K.layers.BatchNormalization(axis=3)(X)
    X = K.layers.Activation('relu')(X)
    
    # --- Second Component of Main Path ---
    # 3x3 convolution
    X = K.layers.Conv2D(
        filters=F3,
        kernel_size=(3, 3),
        padding='same',
        kernel_initializer=init
    )(X)
    X = K.layers.BatchNormalization(axis=3)(X)
    X = K.layers.Activation('relu')(X)
    
    # --- Third Component of Main Path ---
    # 1x1 convolution to restore dimensionality
    X = K.layers.Conv2D(
        filters=F12,
        kernel_size=(1, 1),
        padding='same',
        kernel_initializer=init
    )(X)
    X = K.layers.BatchNormalization(axis=3)(X)
    
    # --- Final Step ---
    # Add Shortcut (Identity) to the main path
    X = K.layers.Add()([X, A_prev])
    X = K.layers.Activation('relu')(X)
    
    return X
