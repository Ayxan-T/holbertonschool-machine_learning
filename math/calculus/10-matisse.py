#!/usr/bin/env python3
"""
Module: 10-mattise
This module is a part of 'Calculus' project.
"""


def poly_derivative(poly):
    
    """
    Docstring for '10-mattise' function.
    """

    if type(poly) is not list:
        return None

    res = []
    for i in range(1, len(poly)):
        res.append(poly[i] * i)

    for item in res:
        if item:
            return res

    # if len(res) == 0:
    return None

    # return [0]
