#!/usr/bin/env python3
"""
    Module: 12-hierarchy
    This module is a part of 'Pandas' project.
"""

import pandas as pd
index = __import__('11-concat').concat


def hierarchy(df1, df2):
    """
        Rearranges the hierarchy of the output of index().
    """

    df = concat(df1.loc[1417411980:1417417980], df2.loc[1417411980:1417417980])

    df = df.swaplevel(0, 1).sort_index()
