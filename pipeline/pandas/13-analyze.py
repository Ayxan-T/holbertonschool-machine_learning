#!/usr/bin/env python3
"""
    Module: 13-analyze
    This module is a part of 'Pandas' project.
"""

import pandas as pd
index = __import__('10-index').index


def analyze(df):
    """
        Return descriptive analytics of df DataFrame.
    """

    return df.describe()
