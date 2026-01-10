#!/usr/bin/env python3
"""
Task 0: From Numpy
"""
import pandas as pd


def from_numpy(array):
    """
    Creates a pd.DataFrame from a np.ndarray
    Args:
        array: np.ndarray
    Returns:
        The newly created pd.DataFrame
    """
    # Generate alphabetical column labels: A, B, C...
    cols = [chr(65 + i) for i in range(array.shape[1])]
    return pd.DataFrame(array, columns=cols)
