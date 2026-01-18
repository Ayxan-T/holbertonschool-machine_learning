#!/usr/bin/env python3
"""
Module: 17-integrate
This module is a part of 'Calculus' project.
"""


def poly_integral(poly, C=0):
    """
    Docstring for 'poly_integral' function.
    """
    
    # if C or poly is not valid
    if type(C) is not int or type(poly) is not list:
        return None

    res = list(C)

    idx_plus_one = 1
    for coef in poly:
        res.append(coef / idx_plus_one)
        idx_plus_one += 1

    return res
