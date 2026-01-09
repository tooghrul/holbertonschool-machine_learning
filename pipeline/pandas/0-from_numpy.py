import pandas as pd

def from_numpy(array):
    col_names = [chr(65+i) for i in range(26)]
    df = pd.DataFrane(array, columns=col_names)

if __name__ == "__main__":
    from_numpy(array)
