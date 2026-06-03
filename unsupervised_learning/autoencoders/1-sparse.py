#!/usr/bin/env python3
""" This module provides a function that creates a sparse autoencoder.

Function:
    autoencoder(input_dims, hidden_layers, latent_dims, lambtha)
"""

import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims, lambtha):
    """ Creates and returns a sparse autoencoder.

    Args:
        input_dims: an integer containing the dimensions of the model input
        hidden_layers: a list containing the number of nodes
            for each hidden layer
        latent_dims: an integer containing the dimensions of the latent space
            representation
        lambtha: the regularization parameter used for L1 regularization
            on the encoded output

    Returns:
        encoder: the encoder model
        decoder: the decoder model
        auto: the sparse autoencoder model
    """

    # IMPLEMENTING ENCODER
    encoder_input = keras.layers.Input(shape=(input_dims,))
    x = encoder_input

    # Add hidden layers
    for num_neurons in hidden_layers:
        x = keras.layers.Dense(num_neurons, activation="relu")(x)

    encoder_output = keras.layers.Dense(
        latent_dims,
        activation="relu",
        activity_regularizer=keras.regularizers.l1(lambtha)
    )(x)

    encoder = keras.models.Model(inputs=encoder_input, outputs=encoder_output)

    # IMPLEMENTING DECODER
    decoder_input = keras.layers.Input(shape=(latent_dims, ))
    x = decoder_input

    # Add hidden layers in reversed order
    for num_neurons in reversed(hidden_layers):
        x = keras.layers.Dense(num_neurons, activation="relu")(x)

    decoder_output = keras.layers.Dense(input_dims, activation="sigmoid")(x)

    decoder = keras.models.Model(inputs=decoder_input, outputs=decoder_output)

    # CONSTRUCTING AUTOENCODER
    auto_input = keras.layers.Input(shape=(input_dims,))
    h = encoder(auto_input)
    auto_output = decoder(h)

    auto = keras.models.Model(inputs=auto_input, outputs=auto_output)

    auto.compile(optimizer="adam", loss="binary_crossentropy")

    return encoder, decoder, auto
