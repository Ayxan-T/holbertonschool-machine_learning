#!/usr/bin/env python3
"""
Module: 0-likelihood
"""


import numpy as np


def likelihood(x, n, P):
    """
    Function: likelihood
    """
    if type(n) is not int or  n <= 0:
        raise ValueError("n must be a positive integer")

    if type(x) is not int or x < 0:
        raise ValueError("x must be an integer that is greater than or equal to 0")

    if x > n:
        raise ValueError("x cannot be greater than n")

    if type(P) is not np.ndarray or P.ndim != 1:
        raise TypeError("P must be a 1D numpy.ndarray")

    for elm in P:
        if elm < 0 or elm > 1:
            raise ValueError("All values in P must be in the range [0, 1]")

    combinations = np.math.factorial(n) / np.math.factorial(n - x) * np.math.factorial(x)

    prob = combinations * (P**x) * ((1 - P)**(n - x))
    
    return prob    
