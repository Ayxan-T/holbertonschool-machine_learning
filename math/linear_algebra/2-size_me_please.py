#!/usr/bin/env python3
"""
Module: 2-size_me_please
This module is a part of 'Linear Algebra' project.
"""


def matrix_shape(matrix):
    """
    Given the nested list of lists, the function returns the shape of the 'matrix'.
    """
    size = [len(matrix)]

    while True:
        try:
            matrix = matrix[0]
        except:
            break

        size.append(len(matrix))
