#!/usr/bin/env python3
"""
Docstring for '101-the_whole_barn' module
"""

import numpy as np


def add_matrices(mat1, mat2):
    """
    Docstring for 'add_matrices(mat1, mat2)' function
    """
    shape_mat1 = numpy.array(mat1).shape
    shape_mat2 = numpy.array(mat2).shape

    for mat1_dim, mat2_dim in zip(shape_mat1, shape_mat2):
        if mat1_dim != mat2_dim:
            return None

    return mat1 + mat2
