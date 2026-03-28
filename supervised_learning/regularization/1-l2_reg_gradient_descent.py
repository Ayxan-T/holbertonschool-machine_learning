#!/usr/bin/env python3
"""
Module: 1-l2_reg_gradient_descent
"""

import numpy as np


def l2_reg_gradient_descent(Y, weights, cache, alpha, lambtha, L):
    """
    Function: l2_reg_gradient_descent

    Y<#classes, m>,
    weights['Wn']<#neurons, #Ws>, weights['bn']<#neurons, 1>, 
    cache['An']<#Ws, m>

    """
    m = cache["A0"].shape[1]
    # print(m)
    print(cache['A' + str(L)].shape)
    print(Y.shape)

    grads = dict()  # dW's, db's

    # Calculate last layer's output (before activation) grad
    dZ_cache = Y * (cache['A' + str(L)] - 1) + (1 - Y) * cache['A' + str(L)]

    grads["dW" + str(L)] = cache['A' + str(L - 1)] * dZ_cache + (lambtha / m) * weights["W" + str(L)] 
    grads["db" + str(L)] = dZ_cache

    for l in range(L-1, 0, -1):
        # Calculate dZ
        dZ_cache = dZ_cache * weights["W" + str(l+1)] * (1 - np.square(cache["A" + str(l)]))

        grads["dW" + str(l)] = np.average(dZ_cache * cache["A" + str(l-1)] + (lambtha / m) * weights["W" + str(l)], axis=1) 
        grads["db" + str(l)] = np.average(dZ_cache, axis=1)

    for l in range(1, L):
        weights["W" + str(l)] -= alpha * grads["dW" + str(l)]
        weights["b" + str(l)] -= alpha * grads["db" + str(l)]
