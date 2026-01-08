#!/usr/bin/env python3
"""
    Module: 5-slice
    This module is a part of 'Pandas' project.
"""


def flip_switch(df):
    """
        Sorts in reverse chronological order and transposes.
    """

    df.sort_values(by='Timestamp', ascending=False, inplace=True)
    return df.T
