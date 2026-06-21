#!/usr/bin/env python3
"""4-deep_rnn module

Provides a function that performs forward propagation for a deep RNN."""

import numpy as np


def deep_rnn(rnn_cells, X, h_0):
    """Performs forward propagation for a deep RNN.

    Args:
        rnn_cells: list of RNNCell instances
        X: np.ndarray(t, m, i) -the data to be used
        h_0: np.ndarray(l, m, h) -initial hidden state

    Returns:
        H: numpy.ndarray -all of the hidden states
        Y: numpy.ndarray -all of the outputs
    """
    t, m, _ = X.shape
    leng = len(rnn_cells)
    _, _, h = h_0.shape
    o = rnn_cells[-1].Wy.shape[1]

    H = np.zeros((t + 1, leng, m, h))
    Y = np.zeros((t, m, o))
    H[0] = h_0

    for time_step in range(t):
        layer_input = X[time_step]

        for layer in range(leng):
            h_prev = H[time_step, layer]
            h_next, y = rnn_cells[layer].forward(h_prev, layer_input)
            H[time_step + 1, layer] = h_next
            layer_input = h_next

        Y[time_step] = y

    return H, Y
