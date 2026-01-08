#!/usr/bin/env python3
"""
    Module: 5-slice
    This module is a part of 'Pandas' project.
"""


def slice(df):
    """
        Selects every 60th row in the Data Frame.
    """
    
    sliced_df = df[['High', 'Low', 'Close', 'Volume_BTC']][::60]
    sliced_df = sliced_df.rename(columns={'Volume_BTC': 'Volume_(BTC)'})
    return sliced_df

