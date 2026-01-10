#!/usr/bin/env python3
import matplotlib.pyplot as plt
import pandas as pd
from_file = __import__('2-from_file').from_file

df = from_file('coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv', ',')

# Remove Weighted_Price
df = df.drop(columns=['Weighted_Price'])

# Rename Timestamp to Date
df = df.rename(columns={'Timestamp': 'Date'})

# Convert Timestamp to date values and index
df['Date'] = pd.to_datetime(df['Date'], unit='s')
df = df.set_index('Date')

# Fill missing Close with previous row value
df['Close'].fillna(method='ffill', inplace=True)

# Fill missing High, Low, Open with corresponding Close value
for col in ['High', 'Low', 'Open']:
    df[col].fillna(df['Close'], inplace=True)

# Fill missing Volume with 0
df['Volume_(BTC)'].fillna(0, inplace=True)
df['Volume_(Currency)'].fillna(0, inplace=True)

# Filter data from 2017 and beyond
df = df[df.index.year >= 2017]

# Resample to daily intervals with specific aggregations
df = df.resample('D').agg({
    'High': 'max',
    'Low': 'min',
    'Open': 'mean',
    'Close': 'mean',
    'Volume_(BTC)': 'sum',
    'Volume_(Currency)': 'sum'
})

# Plot the data
df.plot()
plt.show()

# Print the transformed dataframe as seen in the example output
print(df)
