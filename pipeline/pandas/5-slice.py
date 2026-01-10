#!/usr/bin/env python3
"""
Task 5: Slice
"""


def slice(df):
    """
    Extracts columns High, Low, Close, Volume_(BTC) and
    selects every 60th row.
    Args:
        df: pd.DataFrame
    Returns:
        The sliced pd.DataFrame
    """
    cols = ['High', 'Low', 'Close', 'Volume_(BTC)']
    return df[cols].iloc[::60]
