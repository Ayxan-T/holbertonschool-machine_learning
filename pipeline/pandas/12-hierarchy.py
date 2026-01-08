#!/usr/bin/env python3
"""
    Module: 12-hierarchy
    This module is a part of 'Pandas' project.
"""

import pandas as pd
index = __import__('10-index').index


def hierarchy(df1, df2):
    """
        Rearranges the hierarchy of the output of index().
    """

    df1 = df1.set_index(df1)
    df2 = df2.set_index(df2)
    
    df = pd.concat([df2.loc[1417411980:1417417980], df1.loc[1417411980:1417417980]], keys=['bitstamp', 'coinbase'])

    df = df.swaplevel(0, 1).sort_index()

    return df
    

