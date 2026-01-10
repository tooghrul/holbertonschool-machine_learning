#!/usr/bin/env python3
"""
Task 7: Sort
"""


def high(df):
    """
    Sorts the dataframe by the High price in descending order.
    Args:
        df: pd.DataFrame
    Returns:
        The sorted pd.DataFrame
    """
    return df.sort_values('High', ascending=False)
