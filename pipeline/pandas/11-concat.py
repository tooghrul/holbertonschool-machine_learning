#!/usr/bin/env python3
"""
Task 11: Concat
"""
index = __import__('10-index').index


def concat(df1, df2):
    """
    Concatenates two dataframes with specific indexing and keys.
    Args:
        df1: pd.DataFrame (coinbase)
        df2: pd.DataFrame (bitstamp)
    Returns:
        The concatenated pd.DataFrame
    """
    df1 = index(df1)
    df2 = index(df2)

    # Select df2 up to and including 1417411920
    df2_sub = df2.loc[:1417411920]

    # Concatenate df2 (top) and df1 (bottom) with keys
    return pd.concat([df2_sub, df1], keys=['bitstamp', 'coinbase'])
