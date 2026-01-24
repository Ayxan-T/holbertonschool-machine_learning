#!/usr/bin/env python3
"""
Docstring for '102-squashed_like_sardines' module
"""


def cat_matrices(mat1, mat2, axis=0):
    """
    Docstring for 'cat_matrices' function
    """
    def cat_recur(mat1, mat2, cur_axis=0):
        if cur_axis == axis:
            return mat1 + mat2
        else:
            return [cat_recur(row_mat1, row_mat2, cur_axis+1)
                    for row_mat1, row_mat2 in zip(mat1, mat2)]

    shape = []

    cur_axis = 0
    m1, m2 = mat1, mat2
    while True:
        try:
            if len(m1) != len(m2) and cur_axis != axis:
                return None
            else:
                shape.append(len(m1))
        except TypeError:
            break

        m1, m2 = m1[0], m2[0]
        cur_axis += 1

    return cat_recur(mat1, mat2)
