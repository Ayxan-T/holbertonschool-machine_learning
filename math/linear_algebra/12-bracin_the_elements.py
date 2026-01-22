#!/usr/bin/env python3
"""
Docstring for '12-bracin_the_elements' module
"""


def np_elementwise(mat1, mat2):
    """
    Docstring for 'np_elementwise' function
    """
    Sum = mat1 + mat2
    Diff = mat1 - mat2
    Prod = mat1 * mat2
    Div = mat1 / mat2

    return (Sum, Diff, Prod, Div)
