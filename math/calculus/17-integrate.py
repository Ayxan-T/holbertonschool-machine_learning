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

    res = [C]

    # if poly is 0
    if len(poly) == 1 and poly[0] == 0:
        return res

    idx_plus_one = 1
    for coef in poly:
        new_coef = coef / idx_plus_one

        # if new_coef has no fraction part, save it as an int
        if new_coef - int(new_coef) == 0:
            new_coef = int(new_coef)

        res.append(new_coef)
        
        idx_plus_one += 1

    return res
