#!/usr/bin/env python3
"""
Docstring for '101-the_whole_barn' module
"""


def add_matrices(mat1, mat2):
    """
    Docstring for 'add_matrices(mat1, mat2)' function
    """
    shape = []

    m1, m2 = mat1, mat2
    while True:
        try:
            if len(m1) != len(m2):
                return None
            else:
                shape.append(len(m1))
        except TypeError:
            break

        m1, m2 = m1[0], m2[0]
    
    return add_recur(mat1, mat2);

    
def add_recur(mat1, mat2):
    """
    Helper function: adds two matrices recursively
    """
    mat = []

    # if two lists of ints/floats
    if type(mat1[0]) is not list:
        for elm1, elm2 in zip(mat1, mat2):
            mat.append(elm1 + elm2)
    else:
        for row1, row2 in zip(mat1, mat2):
            mat.append(add_recur(row1, row2))
    return mat
