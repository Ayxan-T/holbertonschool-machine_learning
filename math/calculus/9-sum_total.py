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

    Sum = 0
    for i in range(1, n + 1):
        Sum += i ** 2

    return Sum

