#!/usr/bin/env python3
"""
Module to build a modified LeNet-5 architecture using Keras.
"""
from tensorflow import keras as K


def lenet5(X):
    """
    Builds a modified LeNet-5 architecture using keras.

    Args:
        X: K.Input of shape (m, 28, 28, 1) containing the input images.

    Returns:
        A K.Model compiled with Adam optimizer and accuracy metrics.
    """
    # He Normal initializer with seed 0
    initializer = K.initializers.HeNormal(seed=0)

    # Layer 1: Conv 5x5, 6 filters, same padding, ReLU
    conv1 = K.layers.Conv2D(
        filters=6,
        kernel_size=(5, 5),
        padding='same',
        activation='relu',
        kernel_initializer=initializer
    )(X)

    # Layer 2: Max Pooling 2x2, stride 2x2
    pool1 = K.layers.MaxPooling2D(
        pool_size=(2, 2),
        strides=(2, 2)
    )(conv1)

    # Layer 3: Conv 5x5, 16 filters, valid padding, ReLU
    conv2 = K.layers.Conv2D(
        filters=16,
        kernel_size=(5, 5),
        padding='valid',
        activation='relu',
        kernel_initializer=initializer
    )(pool1)

    # Layer 4: Max Pooling 2x2, stride 2x2
    pool2 = K.layers.MaxPooling2D(
        pool_size=(2, 2),
        strides=(2, 2)
    )(conv2)

    # Flatten the output for the Fully Connected layers
    flatten = K.layers.Flatten()(pool2)

    # Layer 5: Fully Connected, 120 nodes, ReLU
    fc1 = K.layers.Dense(
        units=120,
        activation='relu',
        kernel_initializer=initializer
    )(flatten)

    # Layer 6: Fully Connected, 84 nodes, ReLU
    fc2 = K.layers.Dense(
        units=84,
        activation='relu',
        kernel_initializer=initializer
    )(fc1)

    # Layer 7: Output Fully Connected, 10 nodes, Softmax
    output = K.layers.Dense(
        units=10,
        activation='softmax',
        kernel_initializer=initializer
    )(fc2)

    # Define the model
    model = K.Model(inputs=X, outputs=output)

    # Compile the model with Adam and accuracy metric
    model.compile(
        optimizer=K.optimizers.Adam(),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model
