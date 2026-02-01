#!/usr/bin/env python3
"""
Docstring for '3-adjugate' module
"""


def adjugate(matrix):
    """
    Docstring for 'adjugate' function
    """
    if not isinstance(matrix, list):
        raise TypeError("matrix must be a list of lists")

    if len(matrix) == 0:
        raise TypeError("matrix must be a list of lists")
    
    for row in matrix:
        if not isinstance(row, list):
            raise TypeError("matrix must be a list of lists")
