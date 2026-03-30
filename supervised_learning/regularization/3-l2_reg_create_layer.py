#!/usr/bin/env python3
""" Module: 3-l2_reg_create_layer """

import tensorflow as tf

def l2_reg_create_layer(prev, n, activation, lambtha):
    """
    prev - a tensor containing the output of the previous layer
    n - the number of nodes the new layer should contain
    activation - the activation function that should be used on the layer
    lambtha - the L2 regularization parameter
    Returns: the output of the new layer 
    """
    regularizer = tf.keras.regularizers.L2(lambtha)
    layer = tf.keras.layers.Dense(n, activation=activation, kernel_regularizer=regularizer)(prev)
    return layer
