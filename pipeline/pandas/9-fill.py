#!/usr/bin/env python3
"""
    Module: 9-fill
    This module is a part of 'Pandas' project.
"""


def fill(df):
    """
        Fills the missing values.
    """

    columns_to_drop = ['Weighted_Price']
    df.drop(columns=columns_to_drop, axis=1, inplace=True)

    df['Close'].ffill(inplace=True)

    df['High'] = df['High'].fillna(df['Close'])
    df['Low'] = df['Low'].fillna(df['Close'])
    df['Open'] = df['Open'].fillna(df['Close'])

    df['Volume_(BTC)'] = df['Volume_(BTC)'].fillna(0)
    df['Volume_(Currency)'] = df['Volume_(Currency)'].fillna(0)
