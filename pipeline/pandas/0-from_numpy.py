#!/usr/bin/env python3
import pandas as pd

def from_numpy(array):
    col_count = array.shape()
    col_names = [chr(65+i) for i in range(col_count)]
    df = pd.DataFrame(array, columns=col_names)
    return df
