#!/usr/bin/env python3
"""
Module: 2-precision
"""

import numpy as np


def precision(confusion):
    """
    Function: precision
    """

    diag_nums = np.diag(confusion)
    sums_across_rows = np.sum(confusion, axis=0)

    precisions = diag_nums / sums_across_rows
    return precisions
