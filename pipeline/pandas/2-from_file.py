#!/usr/bin/env python3
"""
task 2 - load from File
"""
import pandas as pd


def from_file(filename, delimiter):
    """
    loads data from a file as a pd.DataFrame
    args:
        filename: file to load from
        delimiter: column separator
    returns:
        The loaded pd.DataFrame
    """
    return pd.read_csv(filename, sep=delimiter)
