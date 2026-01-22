#!/usr/bin/env python3
"""
Docstring for '13-cats_got_your_tongue' module
"""

import numpy as np


def np_cat(mat1, mat2, axis=0):
    return np.concatenate((mat1, mat2), axis=axis)
