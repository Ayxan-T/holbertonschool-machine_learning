#!/usr/bin/env python3
"""1-rnn module

Provides a function that performs forward propagation for a simple RNN."""

import numpy as np


def rnn(rnn_cell, X, h_0):
    """Performs forward propagation for a simple RNN.
    
    Args:
        rnn_cell: RNNCell
        X: np.ndarray(t, m, i) -the data to be used
        h_0: np.ndarray(m, h) -initial state

    Returns:
        H: numpy.ndarray -all the hidden states
        Y: numps.ndarray -all the outputs
    """

    H = [h_0]
    Y = []

    for i, input in enumerate(X):
        h_new, y_new = rnn_cell.forward(H[i], input)
        H.append(h_new)
        Y.append(y_new)

    # H.pop(0)

    return np.array(H), np.array(Y)