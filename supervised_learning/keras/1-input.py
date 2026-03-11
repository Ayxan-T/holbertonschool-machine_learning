#!/usr/bin/env python3
"""
Module: 1-input
"""


import tensorflow.keras as K

def build_model(nx, layers, activations, lambtha, keep_prob):
    # 1. Start the chain
    inputs = K.layers.Input(shape=(nx,))
    reg = K.regularizers.l2(lambtha)

    # 2. 'x' represents the "current" state of the data
    x = inputs

    for i in range(len(layers)):
        # Pass 'x' into the new layer and update 'x' to be the result
        x = K.layers.Dense(
            layers[i],
            activation=activations[i],
            kernel_regularizer=reg
        )(x)

        # Add Dropout to hidden layers (not the output layer)
        if i < len(layers) - 1:
            x = K.layers.Dropout(1 - keep_prob)(x)

    # 3. Keras traces 'x' back to 'inputs' automatically
    model = K.models.Model(inputs=inputs, outputs=x)

    return model
