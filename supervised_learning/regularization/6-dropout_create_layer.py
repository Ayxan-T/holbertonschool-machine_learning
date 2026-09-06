#!/usr/bin/env python3
"""Module: 6-dropout_create_layer"""

import tensorflow as tf


def dropout_create_layer(prev, n, activation, keep_prob, training=True):
    """Creates a layer of a neural network using dropout."""
    layer = tf.keras.layers.Dense(
        units=n,
        activation=activation
    )

    tensor = layer(prev)

    drop_rate = 1 - keep_prob
    dropout = tf.keras.layers.Dropout(rate=drop_rate)
    tensor = dropout(tensor, training=training)

    return tensor
