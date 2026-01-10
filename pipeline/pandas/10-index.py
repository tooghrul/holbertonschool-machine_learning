#!/usr/bin/env python3
"""
Task 10: Indexing
"""


def index(df):
    """
    Sets the Timestamp column as the index of the dataframe.
    Args:
        df: pd.DataFrame
    Returns:
        The modified pd.DataFrame
    """
    return df.set_index('Timestamp')
