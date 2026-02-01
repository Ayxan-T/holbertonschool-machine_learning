#!/usr/bin/env python3
"""
Docstring for '2-cofactor' module
"""


def cofactor(matrix):
    """
    Docstring for 'cofactor' function
    """
    if not isinstance(matrix, list):
        raise TypeError("matrix must be a list of lists")

    if len(matrix) == 0:
        raise TypeError("matrix must be a list of lists")
    
    for row in matrix:
        if not isinstance(row, list):
            raise TypeError("matrix must be a list of lists")


