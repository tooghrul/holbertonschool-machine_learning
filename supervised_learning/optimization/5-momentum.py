#!/usr/bin/env python3
"""Module to update variables using gradient descent with momentum."""
import numpy as np


def update_variables_momentum(alpha, beta1, var, grad, v):
    """
    Updates a variable using gradient descent with momentum.

    Parameters:
    - alpha: learning rate
    - beta1: momentum weight
    - var: numpy.ndarray containing the variable to be updated
    - grad: numpy.ndarray containing the gradient of var
    - v: numpy.ndarray containing the previous first moment of var

    Returns:
    - var: updated variable
    - v: updated first moment
    """
    # Update biased first moment estimate
    v = beta1 * v + (1 - beta1) * grad

    # Update variable using momentum
    var = var - alpha * v

    return var, v
