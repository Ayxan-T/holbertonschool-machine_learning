#!/usr/bin/env python3
"""0-rnn_cell module

Provides a class that represent a cell of a simple RNN."""

import numpy as np


class RNNCell:

    def __init__(self, i, h, o):
        self.Wh = np.random.normal(size=(i+h, h))
        self.Wy = np.random.normal(size=(h, o))
        self.bh = np.zeros(size=(1, h))
        self.by = np.zeros(size=(1, o))
    
    def forward(self, h_prev, x_t):
        concat = np.concatenate((h_prev, x_t), axis=1)
        h_next = np.tanh(np.matmul(concat, self.Wh) + self.bh)
        y_linear = np.matmul(h_next, self.Wy) + self.by
        exp_y = np.exp(y_linear - np.max(y_linear, axis=1, keepdims=True))
        y = exp_y / np.sum(exp_y, axis=1, keepdims=True)
        return h_next, y
        