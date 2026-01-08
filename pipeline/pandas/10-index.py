#!/usr/bin/env python3
"""
    Module: 10-index
    This module is a part of 'Pandas' project.
"""


def index(df):
    """
        Sets the 'Timestamp' column as index column.
    """
    
    return df.set_index('Timestamp')
    
