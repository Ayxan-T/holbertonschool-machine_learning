#!/usr/bin/env python3
"""
    Module: 4-array
    This module is a part of 'Pandas' project.
"""


def array(df):
    """
        Converts the last 10 rows of a Data Frame into a numpy array.
    """

    return df[["High", "Close"]].tail(10).values
