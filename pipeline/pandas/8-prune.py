#!/usr/bin/env python3
"""
    Module: 7-high
    This module is a part of 'Pandas' project.
"""


def prune(df):
    """
        Drops rows where 'Close' column is NAN.
    """

    return df.dropna(subset=['Close'])
