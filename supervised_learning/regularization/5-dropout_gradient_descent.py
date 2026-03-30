#!/usr/bin/env python3
""" Module: 5-dropout_gradient_descent """

import numpy as np


def dropout_gradient_descent(Y, weights, cache, alpha, keep_prob, L):
    """
    Function: l2_reg_gradient_descent

    Y<#classes, m>,
    weights['Wn']<#neurons, #Ws>, weights['bn']<#neurons, 1>,
    cache['An']<#Neurons, m> for last layer: <#classes, m>

    """
    m = cache["A0"].shape[1]

    grads = dict()  # dW's, db's

    # Apply dropout masks
    for layer in range(1, L):
        cache["A" + layer] *= cache["D" + layer] / keep_prob

    # Calculate last layer's output (before activation) grad
    dZ_cache = Y * (cache['A' + str(L)] - 1) + (1 - Y) * cache['A' + str(L)]

    grads["dW" + str(L)] = np.matmul(dZ_cache, cache['A' + str(L - 1)].T) / m
    grads["db" + str(L)] = np.average(dZ_cache, axis=1, keepdims=True)

    for layer in range(L-1, 0, -1):
        # Calculate dZ
        dZ_cache = np.matmul(weights["W" + str(layer+1)].T,
                             dZ_cache) * \
            (1 - np.square(cache["A" + str(layer)]))

        grads["dW" + str(layer)] = np.matmul(dZ_cache,
                                             cache["A" + str(layer-1)].T) / m
        grads["db" + str(layer)] = np.average(dZ_cache, axis=1, keepdims=True)

    for layer in range(1, L+1):
        weights["W" + str(layer)] -= alpha * grads["dW" + str(layer)]
        weights["b" + str(layer)] -= alpha * grads["db" + str(layer)]
