#!/usr/bin/env python3
"""
Task 12: Hierarchy
"""
import pandas as pd
index = __import__('10-index').index


def hierarchy(df1, df2):
    """
    Rearranges MultiIndex to have Timestamp first, filters by range,
    and concatenates.
    Args:
        df1: pd.DataFrame (coinbase)
        df2: pd.DataFrame (bitstamp)
    Returns:
        The concatenated pd.DataFrame
    """
    df1 = index(df1)
    df2 = index(df2)

    # Filter ranges 1417411980 to 1417417980
    t_start = 1417411980
    t_end = 1417417980
    df1_sub = df1.loc[t_start:t_end]
    df2_sub = df2.loc[t_start:t_end]

    # Concatenate with keys
    df_concat = pd.concat([df2_sub, df1_sub], keys=['bitstamp', 'coinbase'])

    # Swap levels so Timestamp is first (level 0), then source (level 1)
    df_concat = df_concat.swaplevel(0, 1)

    # Sort chronologically (by index)
    return df_concat.sort_index()
