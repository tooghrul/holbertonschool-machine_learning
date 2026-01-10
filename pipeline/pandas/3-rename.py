#!/usr/bin/env python3
"""
task 3 - renaming column
"""
import pandas as pd


def rename(df):
    """
    renames the timestamp column to Datetime, converts to datetime values,
    and selects only Datetime and Close columns.
    args:
        df: pd.DataFrame
    returns:
        The modified pd.DataFrame
    """
    df = df.rename(columns={'Timestamp': 'Datetime'})
    df['Datetime'] = pd.to_datetime(df['Datetime'], unit='s')
    return df[['Datetime', 'Close']]
