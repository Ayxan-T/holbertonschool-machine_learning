#!/usr/bin/env python3
"""
Docstring for '7-gettin_cozy' module.
"""


def cat_matrices2D(mat1, mat2, axis=0):
    """
    Docstring for 'cat_matrices2D' function.
    """

    mat1_shape = [len(mat1), len(mat1[0])]
    mat2_shape = [len(mat2), len(mat2[0])]

    # axis=0 -> vertical concat
    if axis == 0: 
        # if cannot be concat, return None
        if mat1_shape[1] != mat2_shape[1]:
            return None
        else:
            return mat1 + mat2

    # axis=1 -> horizontal concat
    if axis == 1:
        # if cannot be concat, return None
        if mat1_shape[0] != mat2_shape[0]:
            return None
        else:
            # Create a new matrix for horizontal concatenation
            result_matrix = []
            for row_idx in range(mat1_shape[0]):
                result_matrix.append(mat1[row_idx] + mat2[row_idx])
            return result_matrix
