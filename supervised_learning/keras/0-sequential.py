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

    reg = K.regularizers.L2(lamtha)

    model.add(K.layers.Dense(
        layers[0],
        activation=activations[0],
        kernel_regularizer=reg,
        input_shape=(nx,)
    ))

    model.add(K.layers.Dropout(1 - keep_prob))
    
    for i in range(1, len(layers)):
        model.add(K.layers.Dense(
            layers[i],
            activation=activations[i],
            kernel_regularizer=reg,
        ))

        if i < len(layers) - 1:
            model.add(K.layers.Dropout(1 - keep_prob))

    return model

