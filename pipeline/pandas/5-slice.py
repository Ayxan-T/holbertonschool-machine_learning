#!/usr/bin/env python3
"""
    Module: 5-slice
    This module is a part of 'Pandas' project.
"""


def slice(df):
    """
        Selects every 60th row in the Data Frame.
    """
    df = df[['High', 'Low', 'Close', 'Volume_(BTC)']]
    return df[::60]
