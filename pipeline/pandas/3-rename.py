#!/usr/bin/env python3
"""
    Module: 3-rename 
    This module is a part of 'Pandas' project.
"""
import pandas as pd


def rename(df):
    """
        takes a pd.DataFrame as input and performs the following:

    df is a pd.DataFrame containing a column named Timestamp.
    The function should rename the Timestamp column to Datetime.
    Convert the timestamp values to datatime values
    Display only the Datetime and Close column
    Returns: the modified pd.DataFrame
    """
    df.rename(columns={'TimeStamp':'DateTime'}, inplace=True)

    df['DateTime'] = pd.to_datetime(df['DateTime']) 

    df = df[['DateTime', 'Close']]

    return df
