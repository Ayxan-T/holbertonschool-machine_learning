#!/usr/bin/env python3
""" Preprocessing time series data """

import pandas as pd


def preprocess_csvfile():
    """Preprocesses stored csv file and returns saved new file's name.
    
    File being preprocessed: coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09
    """
    df = pd.read_csv("coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv")

    # Forward-fill NAs
    df['Close'] = df['Close'].ffill()
    df['Open'] = df['Open'].fillna(df['Close'])
    df['High'] = df['High'].fillna(df['Close'])
    df['Low'] = df['Low'].fillna(df['Close'])
    df['Weighted_Price'] = df['Weighted_Price'].fillna(df['Close'])
    df['Volume_(BTC)'] = df['Volume_(BTC)'].fillna(0)
    df['Volume_(Currency)'] = df['Volume_(Currency)'].fillna(0)
    print("Forward-filling finished.")

    # Remove earlier big gaps
    non_60_diff_indices = diffs[diffs != 60].index
    last_big_diff_index_in_diffs = non_60_diff_indices[-1]
    df = df.iloc[last_big_diff_index_in_diffs + 1:]
    print(f"Removed data before index {last_big_diff_index_in_diffs + 1} due "
           "to the last big difference.")
    print("No non-60 second differences found after initial filtering. "
           "DataFrame remains unchanged.")
    diffs = df['Timestamp'].diff()
    print("\nUpdated diffs value counts after filtering:")
    print(diffs.value_counts())

    file_name = "coinbase_clean.csv"
    df.to_csv(file_name, index=False)
    return file_name

if __name__ == "__main__":
    preprocess_csvfile()