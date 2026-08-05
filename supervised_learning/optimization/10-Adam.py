#!/usr/bin/env python3
"""Module: 10-Adam"""

import tensorflow as tf


def create_Adam_op(alpha, beta1, beta2, epsilon):
    """Create and return a TensorFlow Adam optimizer."""
    return tf.keras.optimizers.Adam(
        learning_rate=alpha,
        beta_1=beta1,
        beta_2=beta2,
        epsilon=epsilon
    )
