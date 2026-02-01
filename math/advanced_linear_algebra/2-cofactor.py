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

    size = len(matrix)
    for row in matrix:
        if len(row) != size:
            raise ValueError("matrix must be a non-empty square matrix")

    result = [[0 for _ in range(size)] for _ in range(size)]

    def helper(mat, s):
        if s == 0:
            return 1

        if s == 1:
            return mat[0][0]

        if s == 2:
            return mat[0][0]*mat[1][1] - mat[0][1]*mat[1][0]

        det = 0
        for idx, elm in enumerate(mat[0]):
            k = (-1)**idx
            minor = [row[:idx] + row[idx+1:] for row in mat[1:]]
            det += k*elm*helper(minor, s-1)
        return det

    for row_idx, row in enumerate(matrix):
        for col_idx, elm in enumerate(row):
            minor = [row[:col_idx] + row[col_idx+1:]
                     for row in matrix[:row_idx] + matrix[row_idx+1:]]
            result[row_idx][col_idx] = ((-1)**(row_idx + col_idx))*helper(minor, size-1)

    return result 
