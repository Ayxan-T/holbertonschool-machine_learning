#!/usr/bin/env python3
"""
Module: 2-precision
"""

import numpy as np


def precision(confusion):
    """
    Function: precision
    """

    precisions = []
    for col_idx in range(confusion.shape[1]):
        prec = confusion[col_idx, col_idx] / np.sum(confusion[:, col_idx])
        precisions.append(prec)

    return numpy.array(precisions)
