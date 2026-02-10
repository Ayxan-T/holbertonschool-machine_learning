#!/usr/bin/env python3
"""
Module: 0-likelihood
"""


import numpy as np


def likelihood(x, n, P):
    """
    Function: likelihood
    """
    if n <= 0:
        raise ValueError("n must be a positive integer")

    if x < 0:
        raise ValueError("x must be an integer that is greater than or equal to 0")

    if x > n:
        raise ValueError("x cannot be greater than n")

    if type(P) is not numpy.ndarray or P.ndim != 1:
        raise ValueError("P must be a 1D numpy.ndarray")

    for elm in P:
        if elm < 0 or elm > 1:
            raise ValueError("All values in P must be in the range [0, 1]")
