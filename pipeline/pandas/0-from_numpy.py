#!/usr/bin/env python3
"""
    Module: 0-from_numpy
    This module is a part of 'Pandas' project.
"""
import pandas as pd
# import string

def from_numpy(array):
    """
    input:
        array -- numpy.ndarray
    returns:
        df -- pandas.DataFrame
    """

    cols_num = array.shape[1]
    # cols = list(string.ascii_uppercase[:cols_num])
    
    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    cols = list(alphabet[:cols_num])
    
    return pd.DataFrame(array, columns=cols)
