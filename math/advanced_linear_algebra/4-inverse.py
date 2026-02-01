#!/usr/bin/env python3
"""
Docstring for '4-inverse' module
"""


def inverse(matrix):
    """
    Docstring for 'inverse' function
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

    det = helper(matrix, size)

    if det == 0:
        return None

    adjugate = [[0 for _ in range(size)] for _ in range(size)]

    for row_idx, row in enumerate(matrix):
        for col_idx, elm in enumerate(row):
            minor = [row[:col_idx] + row[col_idx+1:]
                     for row in matrix[:row_idx] + matrix[row_idx+1:]]
            adjugate[row_idx][col_idx] = ((-1)**(row_idx +
                                          col_idx))*helper(minor, size-1)

    for i in range(size):
        for j in range(i+1, size):
            temp = adjugate[i][j]
            adjugate[i][j] = adjugate[j][i]
            adjugate[j][i] = temp

    for i in range(len(adjugate)):
        for j in range(len(adjugate[i])):
            adjugate[i][j] = adjugate[i][j] / det

    return adjugate
