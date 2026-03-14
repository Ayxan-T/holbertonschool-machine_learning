#!/usr/bin/env python3
"""
Module: 0-norm_constants
"""

import numpy as np


def normalization_constants(X):
    """
    for X<m, nx>,
    returns mean and stddev for each feature
    """
    mean = X.mean(axis=0)
    stdev = X.stdev(axis=0)

    return mean, stdev
