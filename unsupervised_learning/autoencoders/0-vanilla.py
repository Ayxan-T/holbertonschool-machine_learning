#!/usr/bin/env python3
""" Module: 0-vanilla.py """

import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims):
    """
    input_dims is an integer containing the dimensions of the model input
    hidden_layers is a list containing the number of nodes for each hidden layer in the encoder, respectively
    the hidden layers should be reversed for the decoder
    latent_dims is an integer containing the dimensions of the latent space representation
    Returns: encoder, decoder, auto
    encoder is the encoder model
    decoder is the decoder model
    auto is the full autoencoder model
    """

    encoder = keras.models.Sequential()
    decoder = keras.models.Sequential()
    auto = keras.models.Sequential()

    # Encoder path
    encoder.add(keras.layers.Input(shape=(input_dims,)))
    for nodes in hidden_layers:
        encoder.add(keras.layers.Dense(nodes, activation='relu'))
    encoder.add(keras.layers.Dense(latent_dims, activation='relu', name='encoder_output'))

    # Decoder path
    decoder.add(keras.layers.Input(shape=(latent_dims,)))
    for nodes in reversed(hidden_layers):
        decoder.add(keras.layers.Dense(nodes, activation='relu'))
    # The last layer in the decoder should use sigmoid activation
    decoder.add(keras.layers.Dense(input_dims, activation='sigmoid', name='decoder_output'))

    # Autoencoder path (combining encoder and decoder)
    auto_input = keras.layers.Input(shape=(input_dims,))
    encoded = encoder(auto_input)
    decoded = decoder(encoded)
    auto = keras.models.Model(inputs=auto_input, outputs=decoded)

    # Compile the autoencoder model
    auto.compile(optimizer='adam', loss='binary_crossentropy')

    return encoder, decoder, auto