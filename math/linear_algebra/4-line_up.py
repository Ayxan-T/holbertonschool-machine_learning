#!/usr/bin/env python3
"""
Module: 3-line_up
This module is a part of 'Linear Algebra' Project.
"""


def add_arrays(arr1, arr2):
    """
    Docstring for 'add_arrays' function.
    """

    if len(arr1) != len(arr2):
        return None

    res = []
    for a, b in zip(arr1, arr2):
        res.append(a + b)

    return res
