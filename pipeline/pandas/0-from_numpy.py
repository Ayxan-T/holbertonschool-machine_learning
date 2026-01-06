#!/usr/bin/env python3
"""
    Module: 0-from_numpy
    This module is a part of 'Pandas' project.
"""
import pandas as pd


def from_numpy(array):
    """
    input:
        array -- numpy.ndarray
    returns:
        df -- pandas.DataFrame
    """
    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    cols_num = array.shape[1]
    cols = [letter for letter in alphabet[:cols_num]]
    return pd.DataFrame(array, columns = cols)
