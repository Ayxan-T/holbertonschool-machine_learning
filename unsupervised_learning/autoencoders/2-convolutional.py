#!/usr/bin/env python3
""" This module provides a function that creates a convolutional autoencoder.

Function: autoencoder()
"""

import tensorflow.keras as keras


def autoencoder(input_dims, filters, latent_dims):
    """ Creates and returns a convolutional autoencoder.

    Each convolution in the encoder uses a kernel size of (3, 3)
    with same padding and relu activation, followed by max pooling
    of size (2, 2).

    Each convolution in the decoder, except for the last two, uses
    a filter size of (3, 3) with same padding and relu activation,
    followed by upsampling of size (2, 2). The second to last convolution 
    use valid padding. The last convolution should have the same number
    of filters as the number of channels in input_dims with sigmoid activation
    and no upsampling.

    Autoencoder is compiled using Adam optimization and binary cross-entropy
    loss.

    Args:
        input_dims: a tuple of integers containing the dimensions
            of the model input
        filters:  a list containing the number of filters for each
            convolutional layer in the encoder, respectively
        latent_dims:  a tuple of integers containing the dimensions
            of the latent space representation
    
    Returns:
        encoder: the encoder model
        decoder: the decoder model
        auto: the full autoencoder model
    """

    encoder_input = keras.layers.Input(shape=input_dims)
    x = encoder_input

    for num_filters in filters:
        x = keras.layers.Conv2D(
            num_filters, (3, 3), padding='same', activation='relu'
        )(x)
        x = keras.layers.MaxPooling2D((2, 2), padding='same')(x)
    
    encoder_output = x
    encoder = keras.models.Model(encoder_input, encoder_output)

    decoder_input = keras.layers.Input(shape=latent_dims)
    x = decoder_input
    
    for num_filters in reversed(filters[2:]):
        x = keras.layers.Conv2D(
            num_filters, (3, 3), padding='same', activation='relu'
        )(x)
        x = keras.layers.UpSampling2D((2, 2))(x)
    
    # second to last layer
    x = keras.layers.Conv2D(
        filters[1], (3, 3), padding='valid', activation='relu'
    )(x)
    x = keras.layers.UpSampling2D((2, 2))(x)

    # last layer
    decoder_output = keras.layers.Conv2D(
        input_dims[2], (3, 3), padding='same', activation='sigmoid'
    )(x)

    decoder = keras.models.Model(decoder_input, decoder_output)

    auto = keras.Model(encoder_input, decoder(encoder(encoder_input)))
    auto.compile(optimizer='adam', loss='binary_crossentropy')

    return encoder, decoder, auto
