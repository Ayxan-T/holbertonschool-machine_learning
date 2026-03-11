#!/usr/bin/env python3
"""
Module: 1-input
"""

import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    """
    Function: build_model
    """

    input = K.layers.Input(shape=(nx,))

    reg = K.regularizers.l2(lambtha)

    next_layers = []
    for i in range(len(layers)):
        if i == 0:
            next_layers.append(
                K.layers.Dense(
                    layers[i],
                    activation=activations[i],
                    kernel_regularizer=reg
                )(input))
        elif i == 1:
            next_layers.append(
                K.layers.Dense(
                    layers[i],
                    activation=activations[i],
                    kernel_regularizer=reg
                )(next_layers[i - 1]))
            next_layers.append(
                K.layers.Dropout(1 - keep_prob)(next_layers[i])
            )
        else:
            next_layers.append(
                K.layers.Dense(
                    layers[i],
                    activation=activations[i],
                    kernel_regularizer=reg
                )(next_layers[i - 2]))
            next_layers.append(
                K.layers.Dropout(1 - keep_prob)(next_layers[i])
            )

    model = K.models.Model(inputs=input, outputs=next_layers[-1])

    return model


