#!/usr/bin/env python3
"""Save and load Keras models"""
import tensorflow.keras as K


def save_model(network, filename):
    """
    Saves the entire Keras model to a file.

    Parameters:
    - network: Keras model to save
    - filename: path where the model should be saved
    """
    network.save(filename)


def load_model(filename):
    """
    Loads a Keras model from a file.

    Parameters:
    - filename: path to the saved model file

    Returns:
    - The loaded Keras model
    """
    return K.models.load_model(filename)
