#!/usr/bin/env python3
"""
Module: 0-l2_reg_cost
"""

import numpy as np


def l2_reg_cost(cost, lambtha, weights, L, m):
    """
    Function: l2_reg_cost
    """
    L2_reg = 0
    for i in range(L):
        key = "W" + (i + 1)
        L2_reg += weights[key]**2

    return cost / m + lambtha * L2_reg
