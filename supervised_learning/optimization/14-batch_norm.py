#!/usr/bin/env python3
"""Module: 14-batch_norm"""

import tensorflow as tf


def create_batch_norm_layer(prev, n, activation):
    """Creates a batch normalization layer for neural network in tensorflow."""
    # Base dense layer with requested initializer and no internal bias
    dense = tf.keras.layers.Dense(
        units=n,
        kernel_initializer=tf.keras.initializers.VarianceScaling(
            mode="fan_avg"
        ),
        activation=activation
    )(prev)

    # Batch normalization layer applied before activation
    bn = tf.keras.layers.BatchNormalization(
        gamma_initializer=tf.keras.initializers.Ones(),
        beta_initializer=tf.keras.initializers.Zeros(),
        epsilon=1e-7
    )(dense)

    return bn
