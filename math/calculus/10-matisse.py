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

    empty_with_0s = True
    for item in poly:
        if item: 
            empty_with_0s = False
            break

    if empty_with_0s:
        return None
        
    res = []
    for i in range(1, len(poly)):
        res.append(poly[i] * i)

    for item in res:
        if item:
            return res

    if len(res) == 0:
        return None

    return [0]
