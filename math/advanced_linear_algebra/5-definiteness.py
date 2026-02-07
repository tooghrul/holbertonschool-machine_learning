#!/usr/bin/env python3
""" Definiteness of a matrix """
import numpy as np


def definiteness(matrix):
    """ Function for determining definiteness of a matrix """

    if not isinstance(matrix, np.ndarray):
        raise TypeError('matrix must be a numpy.ndarray')

    if matrix.ndim != 2 or (matrix.shape[0] != matrix.shape[1]):
        return None

    if not np.array_equal(matrix, matrix.T):
        return None

    eigenvals = np.linalg.eigvals(matrix)

    if np.all(eigenvals > 0):
        return "Positive definite"
    if np.all(eigenvals < 0):
        return "Negative definite"
    if np.all(eigenvals >= 0):
        return "Positive semi-definite"
    if np.all(eigenvals <= 0):
        return "Negative semi-definite"
    return "Indefinite"
