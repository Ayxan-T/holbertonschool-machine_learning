#!/usr/bin/env python3
"""
Module: 0-Sequential
"""

import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    """
    Function: build_model
    """

    model = K.Sequential()

    reg = K.regularizers.L2(lambtha)

    for i in range(len(layers)):
        if i == 0:
            # The first layer must define the input_shape (nx,)
            model.add(K.layers.Dense(
                layers[i],
                activation=activations[i],
                kernel_regularizer=reg,
                input_shape=(nx,)
            ))
        else:
            model.add(K.layers.Dense(
                layers[i],
                activation=activations[i],
                kernel_regularizer=reg
            ))

        # Add Dropout after every layer except the last (output) layer
        if i < len(layers) - 1:
            model.add(K.layers.Dropout(1 - keep_prob))

    return model
