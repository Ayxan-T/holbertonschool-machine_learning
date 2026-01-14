#!/usr/bin/env python3
"""
Module: 9-sum_total.py
This module is a part of 'Calculus' project.
"""

def summation_i_squared(n):
    """
    Docstring for 'summation_i_squared' function.
    """
    if n < 1:
        return None
    
    try:
        return n ** 2 + summation_i_squared(n - 1)
    except TypeError:
        return n ** 2
