#!/usr/bin/env python3
"""
    Module: 5-slice
    This module is a part of 'Pandas' project.
"""


def high(df):
    """
        Sorts in reverse by 'High' column.
    """

    df.sort_values(by='High', ascending=False, inplace=True)
    return df
