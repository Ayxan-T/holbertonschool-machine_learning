#!/usr/bin/env python3
"""2-gru_cell module

Provides a class that represents a gated recurrent unit cell."""

import numpy as np


class GRUCell:
	"""A gated recurrent unit cell."""

	def __init__(self, i, h, o):
		"""Initialize the GRU cell weights and biases."""
		self.Wz = np.random.normal(size=(i + h, h))
		self.Wr = np.random.normal(size=(i + h, h))
		self.Wh = np.random.normal(size=(i + h, h))
		self.Wy = np.random.normal(size=(h, o))
		self.bz = np.zeros((1, h))
		self.br = np.zeros((1, h))
		self.bh = np.zeros((1, h))
		self.by = np.zeros((1, o))

	def forward(self, h_prev, x_t):
		"""Perform forward propagation for one time step."""
		concat = np.concatenate((h_prev, x_t), axis=1)

		z = 1 / (1 + np.exp(-(np.matmul(concat, self.Wz) + self.bz)))
		r = 1 / (1 + np.exp(-(np.matmul(concat, self.Wr) + self.br)))

		concat_candidate = np.concatenate((r * h_prev, x_t), axis=1)
		h_hat = np.tanh(np.matmul(concat_candidate, self.Wh) + self.bh)
		h_next = (1 - z) * h_prev + z * h_hat

		y_linear = np.matmul(h_next, self.Wy) + self.by
		y_exp = np.exp(y_linear - np.max(y_linear, axis=1, keepdims=True))
		y = y_exp / np.sum(y_exp, axis=1, keepdims=True)

		return h_next, y
