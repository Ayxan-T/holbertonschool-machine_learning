#!/usr/bin/env python3
"""
Docstring for '5-across_the_planes' module.
"""

import numpy as np


def add_matrices2D(mat1, mat2):
    """
    Docstring for 'add_matrices2D' function.
    """
 
    if np.array(mat1).shape != np.array(mat2).shape:
        return None
