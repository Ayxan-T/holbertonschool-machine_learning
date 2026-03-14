#!/usr/bin/env python3
"""
Module: 2-shuffle_data
"""

import numpy as np


def shuffle_data(X, Y):
    """
    for X<m, nx>, Y<m, ny>,
    returns new X and Y shuffled in the same fashion
    """
    np.random.seed(0)
    return np.random.permutation(X), np.random.permutation(Y)
