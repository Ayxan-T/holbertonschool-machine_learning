#!/usr/bin/env python3
"""
Doctring for '7-riding_bareback' module.
"""


def mat_mul(mat1, mat2):
    """
    Docstring for 'mat_mul' function.
    """

    mat1_shape = [len(mat1), len(mat1[0])]
    mat2_shape = [len(mat2), len(mat2[0])]

    # if cannot be multiplied
    if mat1_shape[1] != mat2_shape[0]:
        return None

    mat = [[] for _ in range(mat1_shape[0])]

    for row_mat1_idx, row_mat1 in enumerate(mat1):
        col_idx = 0
        while col_idx < mat2_shape[1]:
            Sum = 0
            for row_mat2, row_mat1_elm in zip(mat2, row_mat1):
                Sum += row_mat2[col_idx] * row_mat1_elm

            mat[row_mat1_idx].append(Sum)
            col_idx += 1

    return mat
