#!/usr/bin/env python3
"""
    Module: 11-concat
    This module is a part of 'Pandas' project.
"""

import pandas as pd
index = __import__("10-index").index


def concat(df1, df2):
    """
        Concatinates two data frames.
    """

    df1 = index(df1)
    df2 = index(df2)

    df2 = df2[:1417411921]

    df = pd.concat([df2, df1], keys=['bitstamp', 'coinbase'])   

    return df
