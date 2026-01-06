#!/usr/bin/env python3
"""
    Module: 2-from_file
    This module is a part of 'Pandas' project.
"""
import pandas as pd


def from_file(filename, delimiter):
    """
        Reads a CSV file and constructs a Data Frame.
        args:
            filename -- the csv file to be read
            delimiter -- the delimiter used in the csv file
        output:
            a pandas DataFrame object
    """

    return pd.read_csv(filename, sep=delimiter)
