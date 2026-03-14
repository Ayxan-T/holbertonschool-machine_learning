#!/usr/bin/env python3
"""
Module: 3-mini_batch
"""

import numpy as np


def create_mini_batches(X, Y, batch_size):
    """
    for X<m, nx>, Y<m, ny>,
    returns mini-batches of size batch_size
    """
    mini_batches = []
    num_batches = np.ceil(X.shape[0] / batch_size)
    for i in range(int(num_batches)):
        start = i * batch_size
        end = (i + 1) * batch_size
        mini_batch_X = X[start:end]
        mini_batch_Y = Y[start:end]
        mini_batches.append((mini_batch_X, mini_batch_Y))
    return mini_batches
