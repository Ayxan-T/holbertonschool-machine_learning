#!/usr/bin/env python3
"""
Module: 1-sensitivity
"""

import numpy as np


def sensitivity(confusion):
    """
    Function: sensitivity
    """

    diag_nums = np.diag(confusion)
    sums_across_cols = np.sum(confusion, axis=1)


    sensitivities = diag_nums / sums_across_cols

    return sensitivities
