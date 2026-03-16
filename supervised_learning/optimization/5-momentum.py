#!/usr/bin/env python3
"""
Module: 5-momentum
"""

import numpy as np


def update_variables_momentum(alpha, beta1, var, grad, v):
    """
    Function: update_variables_momentum
    """
    momentum = beta1 * v + (1 - beta1) * var
    var = var - alpha * momentum
    return var, momentum
