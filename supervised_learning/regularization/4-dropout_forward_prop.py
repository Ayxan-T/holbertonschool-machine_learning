#!/usr/bin/env python3
""" Module: 4-dropout_forward_prop """

import numpy as np


def dropout_forward_prop(X, weights, L, keep_prob):
    """ Function: dropout_forward_prop 
    X.shape = (nx, m)
    W.shape = (kx, nx), k being # of neurons
    """
    cache = { "A0": X }
    for layer in range(1, L):
        # Calculate Z
        Z = np.matmul(weights["W" + str(layer)], cache["A" + str(layer-1)]) + weights["b" + str(layer)]

        # Create Dropout mask and save it
        dropout_mask = np.random.rand(Z.shape[0], Z.shape[1]) < keep_prob
        dropout_mask = dropout_mask.astype(float) / keep_prob
        cache["D" + str(layer)] = dropout_mask

        # Calculate A, apply mask and save it
        A = np.tanh(Z)
        cache["A" + str(layer)] = A * dropout_mask
    
    # Calculate final layer output
    Z = np.matmul(weights["W" + str(L)], cache["A" + str(L-1)]) + weights["b" + str(L)]
    Z_exp = np.exp(Z)
    cache["A" + str(L)] = Z_exp / np.sum(Z_exp, axis=0)

    return cache



