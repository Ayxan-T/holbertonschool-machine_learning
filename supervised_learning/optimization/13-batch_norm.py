#!/usr/bin/env python3
"""Module: 13-batch_norm"""

import numpy as np


def batch_norm(Z, gamma, beta, epsilon):
    """Normalizes an unactivated output of a neural network using batch
    normalization."""
    mean = np.mean(Z, axis=0)  # (1, n)
    var = np.var(Z, axis=0)
    X_norm = (Z - mean) / np.sqrt(var + epsilon)
    out = gamma * X_norm + beta
    return out