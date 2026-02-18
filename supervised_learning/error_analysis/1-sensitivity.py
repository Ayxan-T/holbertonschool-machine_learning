#!/usr/bin/env python3
"""
Module: 1-sensitivity
"""

import numpy as np


def sensitivity(confusion):
    """
    Function: sensitivity
    """

    sensitivities = []
    for idx in range(confusion.shape[0]):
        senst = confusion[idx, idx] / confusion[idx].sum()
        sensitivities.append(senst)

    return np.array(sensitivities)
