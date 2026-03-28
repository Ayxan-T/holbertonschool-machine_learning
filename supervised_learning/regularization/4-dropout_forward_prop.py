#!/usr/bin/env python3
""" Module: 4-dropout_forward_prop """

import numpy as np


def dropout_forward_prop(X, weights, L, keep_prob):
    """ Function: dropout_forward_prop 
    X.shape = (nx, m)
    W.shape = (kx, nx), k being # of neurons
    """
    cache = { "A0": X }
    for i in range(1, L):
        W = weights["W" + str(i)]
        b = weights["b" + str(i)]
        
        # 1. Linear
        Z = np.matmul(W, cache["A" + str(i-1)]) + b
        
        # 2. Activation
        A = np.tanh(Z)
        
        # 3. Mask (Generate exactly here to maintain random state)
        mask = (np.random.rand(A.shape[0], A.shape[1]) < keep_prob).astype(int)
        
        # 4. Scale and Apply
        mask /= keep_prob
        cache["D" + str(i)] = mask
        cache["A" + str(i)] = A * mask
    
    # Calculate final layer output
    Z = np.matmul(weights["W" + str(L)], cache["A" + str(L-1)]) + weights["b" + str(L)]
    Z_exp = np.exp(Z)
    cache["A" + str(L)] = Z_exp / np.sum(Z_exp, axis=0, keepdims=True)

    return cache



