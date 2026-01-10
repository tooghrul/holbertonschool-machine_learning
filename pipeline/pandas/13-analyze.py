#!/usr/bin/env python3
"""
Task 13: Analyze
"""


def analyze(df):
    """
    Computes descriptive statistics for all columns except Timestamp.
    Args:
        df: pd.DataFrame
    Returns:
        pd.DataFrame containing statistics
    """
    # Drop Timestamp if it exists as a column (not index)
    if 'Timestamp' in df.columns:
        df = df.drop(columns=['Timestamp'])
    return df.describe()
