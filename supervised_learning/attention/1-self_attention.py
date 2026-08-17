#!/usr/bin/env python3
"""Module: 1-self_attention"""

import tensorflow as tf


class SelfAttention(tf.keras.layers.Layer):
    def __init__(self, units):
        super().__init__()
        self.units = units
        self.W = tf.keras.layers.Dense(units)
        self.U = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, s_prev, hidden_states):
        """Compute the attention context vector and weights."""
        w = self.W(s_prev)
        u = self.U(hidden_states)
        score = self.V(tf.nn.tanh(w + u))
        weights = tf.nn.softmax(score, axis=1)
        context = tf.reduce_sum(weights * hidden_states, axis=1)
        return context, weights
