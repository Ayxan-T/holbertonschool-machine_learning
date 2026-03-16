#!/usr/bin/env python3
"""
Module: 9-Adam
"""

import numpy as np


def update_variables_Adam(alpha, beta1, beta2, epsilon, var, grad, v, s, t):
    """
    Function: update_variables_Adam
    """
    bias_cor_second = 1 - beta2**t
    momentum_second = beta2 * s + (1 - beta2) * grad**2
    momentum_second_corrected = momentum_second / bias_cor_second

    bias_cor = 1 - beta1**t
    momentum = beta1 * v + (1 - beta1) * grad
    momentum_corrected = momentum / bias_cor

    var = var - alpha * (momentum_corrected /
                         (np.sqrt(momentum_second_corrected) + epsilon))
    return var, momentum, momentum_second
