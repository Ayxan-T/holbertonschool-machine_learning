#!/usr/bin/env python3
"""Module: 14-batch_norm"""

import tensorflow as tf


def create_batch_norm_layer(prev, n, activation):
    """Creates a batch normalization layer for neural network in tensorflow."""
    # Base dense layer with requested initializer and no internal bias
    dense = tf.keras.layers.Dense(
        units=n,
        use_bias=False,
        kernel_initializer=tf.keras.initializers.VarianceScaling(
            mode="fan_avg"
        ),
    )(prev)

    # Batch normalization layer applied before activation
    bn = tf.keras.layers.BatchNormalization(
        gamma_initializer="ones", beta_initializer="zeros", epsilon=1e-7
    )(dense)

    # Apply the activation function if provided
    if activation is not None:
        return activation(bn)
    return bn