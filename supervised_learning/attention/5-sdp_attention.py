#!/usr/bin/env python3
"""Module: 5-sdp_attention"""

import tensorflow as tf


def sdp_attention(Q: tf.Tensor, K: tf.Tensor, V: tf.Tensor, mask=None):
    """
        Q: (..., seq_len_q, dk)
        K: (..., seq_len_v, dk)
        V: (..., seq_len_v, dv)
    """
    dk_root = Q.shape[-1] ** 0.5

    scores = tf.matmul(Q, K, transpose_b=True)  # (..., seq_len_q, seq_len_v)

    # Scaling
    scores = scores / dk_root

    # Masking if needed
    if mask is not None:
        mask = mask * -1e9
        scores = scores + mask

    # Softmaxing
    weights = tf.nn.softmax(scores, axis=-1)

    output = tf.matmul(weights, V)

    return output, weights
