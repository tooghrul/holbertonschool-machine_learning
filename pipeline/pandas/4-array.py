#!/usr/bin/env python3
"""
Task 4: To Array
"""
import pandas as pd


def array(df):
    """
    Takes the last 10 rows of the High and Close columns
    and converts them into a numpy.ndarray.
    Args:
        df: pd.DataFrame
    Returns:
        The numpy.ndarray
    """
    return numpy.ndarray(buffer = df[['High', 'Close']].tail(10))
