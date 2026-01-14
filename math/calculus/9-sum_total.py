#!/usr/bin/env python3
"""
Module: 9-sum_total.py
This module is a part of 'Calculus' project.
"""

import numpy as np


def summation_i_squared(n):
    """
    Docstring for 'summation_i_squared' function.
    """
    if n < 1:
        return None

    arr = np.arange(1, n+1)
    arr = arr ** 2
    Sum = np.sum(arr)

    return Sum
