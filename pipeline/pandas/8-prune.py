#!/usr/bin/env python3
"""
Task 8: Prune
"""


def prune(df):
    """
    Removes any entries where Close has NaN values.
    Args:
        df: pd.DataFrame
    Returns:
        The modified pd.DataFrame
    """
    return df.dropna(subset=['Close'])
