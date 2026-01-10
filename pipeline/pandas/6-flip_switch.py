#!/usr/bin/env python3
"""
Task 6: Flip it and Switch it
"""


def flip_switch(df):
    """
    Sorts the data in reverse chronological order and
    transposes the sorted dataframe.
    Args:
        df: pd.DataFrame
    Returns:
        The transformed pd.DataFrame
    """
    return df.sort_values('Timestamp', ascending=False).transpose()
