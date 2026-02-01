#!/usr/bin/env python3
"""
Docstring for '0-determinant' module
"""


def determinant(matrix):
    """
    Docstring for 'determinant' function
    """
    if type(matrix[0]) is not list:
        raise TypeError("matrix must be a list of lists")

    # if zero matrix ( [[]] )
    if len(matrix[0]) == 0:
        return 0

    size = len(matrix)
    for row in matrix:
        if len(row) != size:
            raise ValueError("matrix must be a square matrix")

    def helper(mat, s):
        if s == 2:
            return mat[0][0]*mat[1][1] - mat[0][1]*mat[1][0]

        det = 0
        for idx, elm in mat[0]:
            k = (-1)**idx
            minor = [row[:idx] + row[idx+1:] for row in mat[1:]]
            det += k*elm*helper(minor, s-1)
        
        return det

    return helper(matrix, size)
