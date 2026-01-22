#!/usr/bin/env python3
"""
Docstring for '100-slice_like_a_ninja' module
"""


def np_slice(matrix, axes={}):
    """
    Docstring for 'np_slice' function
    """

    slicers = []
    for key in sorted(matrix.keys()):
        slicers.append(slice(*matrix[key]))

    return matrix[*slicers]
