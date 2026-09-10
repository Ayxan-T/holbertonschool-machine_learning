#!/usr/bin/env python3
"""Module: 6-dropout_create_layer"""

import tensorflow as tf


def dropout_create_layer(prev, n, activation, keep_prob, training=True):
    """Creates a layer of a neural network using dropout."""
    init = tf.keras.initializers.VarianceScaling(
        scale=2.0,
        mode="fan_avg"
    )

    layer = tf.keras.layers.Dense(
        units=n,
        activation=activation,
        kernel_initializer=init
    )

    tensor = layer(prev)

    drop_rate = 1 - keep_prob
    dropout = tf.keras.layers.Dropout(rate=drop_rate)
    tensor = dropout(tensor, training=training)

    return tensor
