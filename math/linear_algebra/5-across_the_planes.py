#!/usr/bin/env python3
"""
Docstring for '5-across_the_planes' module.
"""


def add_matrices2D(mat1, mat2):
    """
    Docstring for 'add_matrices2D' function.
    """
 
    if len(mat1) != len(mat2) or len(mat1[0]) != len(mat2[0]):
        return None

    mat = [[] for _ in range(len(mat1))]

    k = 0
    for row1, row2 in zip(mat1, mat2):
        for elm1, elm2 in zip(row1, row2):
            mat[k].append(elm1 + elm2)
        k += 1

    return mat
