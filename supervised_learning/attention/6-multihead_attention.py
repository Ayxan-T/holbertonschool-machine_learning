#!/usr/bin/env python3
"""Module: 6-multihead_attention"""

import tensorflow as tf
sdp_attention = __import__('5-sdp_attention').sdp_attention


class MultiHeadAttention(tf.keras.layers.Layer):
    """Performs multi head attention."""
    def __init__(self, dm, h):
        super().__init__()
        self.h = h
        self.dm = dm  # dv*h
        self.depth = dm // h  # dv
        self.Wq = tf.keras.layers.Dense(dm)
        self.Wk = tf.keras.layers.Dense(dm)
        self.Wv = tf.keras.layers.Dense(dm)
        self.linear = tf.keras.layers.Dense(dm)

    def call(self, Q, K, V, mask=None):
        """Call to perform multi head attention with args of Q, K, V."""
        Q = self.Wq(Q)  # (batch, seq_len_q, dm)
        K = self.Wk(K)  # (batch, seq_len_v, dm)
        V = self.Wv(V)  # (batch, seq_len_v, dm)

        # Multi-head splitting: (batch, h, seq_len_q/v, depth)
        batch_size = Q.shape[0]
        Q = tf.transpose(tf.reshape(Q, (batch_size, -1, self.h, tf.cast(
            self.depth, tf.int32))), perm=[0, 2, 1, 3])
        K = tf.transpose(tf.reshape(K, (batch_size, -1, self.h, tf.cast(
            self.depth, tf.int32))), perm=[0, 2, 1, 3])
        V = tf.transpose(tf.reshape(V, (batch_size, -1, self.h, tf.cast(
            self.depth, tf.int32))), perm=[0, 2, 1, 3])

        # (batch, h, seq_len_q, depth), (batch, h, seq_len_q, seq_len_v)
        output, weights = sdp_attention(Q, K, V, mask=mask)

        output = tf.transpose(output, perm=[0, 2, 1, 3])

        # (batch, seq_len_q, dm)
        output = tf.reshape(output, (batch_size, -1, self.dm))
        output = self.linear(output)

        return output, weights
