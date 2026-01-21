#!/usr/bin/env python3
"""
Module: 3-flip_me_over
This module is a part of 'Linear Algebra' project.
"""


def matrix_transpose(matrix):
    """
    Given a list of lists, returns the transpose of that 'matrix'.
    """
    matrix_T = [[] for _ in range(len(matrix[0]))]

    for row in matrix:
        for idx, elm in enumerate(row):
            matrix_T[idx].append(elm)

    return matrix_T
