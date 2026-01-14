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

    for i in range(len(poly)):
        poly[i] = poly[i] * i

    for item in poly:
        if item:
            return poly

    return [0]
