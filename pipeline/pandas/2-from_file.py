#!/usr/bin/env python3
"""
    Module: 2-from_file
    This module is a part of 'Pandas' project.
"""
import pandas as pd


def from_file(filename, delimiter):

    return pd.DataFrame(filename, sep=delimiter)
