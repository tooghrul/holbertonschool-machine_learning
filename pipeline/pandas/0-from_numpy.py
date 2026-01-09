#!/usr/bin/env python3
import pandas as pd

def from_numpy(array):
    num_cols = array.shape[1]
    col_names = [chr(65 + i) for i in range(num_cols)]
    return pd.DataFrame(array, columns=col_names)