#!/usr/bin/env python3
"""
Task 3: Rename
"""
import pandas as pd


def rename(df):
    """
    Renames the Timestamp column to Datetime, converts to datetime values,
    and selects only Datetime and Close columns.
    Args:
        df: pd.DataFrame
    Returns:
        The modified pd.DataFrame
    """
    df = df.rename(columns={'Timestamp': 'Datetime'})
    df['Datetime'] = pd.to_datetime(df['Datetime'], unit='s')
    return df[['Datetime', 'Close']]
