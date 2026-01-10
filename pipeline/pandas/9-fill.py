#!/usr/bin/env python3
"""
Task 9: Fill
"""


def fill(df):
    """
    Fills missing values in Close with previous row value.
    Fills High, Low, Open with the Close value of the same row.
    Sets missing Volume values to 0. Removes Weighted_Price.
    Args:
        df: pd.DataFrame
    Returns:
        The modified pd.DataFrame
    """
    # Remove Weighted_Price
    df = df.drop(columns=['Weighted_Price'])

    # Fill Close with previous value (forward fill)
    df['Close'].fillna(method='ffill', inplace=True)

    # Fill High, Low, Open with their row's Close value
    for col in ['High', 'Low', 'Open']:
        df[col].fillna(df['Close'], inplace=True)

    # Set missing Volume to 0
    df['Volume_(BTC)'].fillna(0, inplace=True)
    df['Volume_(Currency)'].fillna(0, inplace=True)

    return df
