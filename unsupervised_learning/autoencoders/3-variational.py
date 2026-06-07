#!/usr/bin/env python3
""" This module provides a function which creates a variational autoencoder.

Function: 
"""

import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims):
    """ Creates a variational autoencoder.
    
    Args: 
        input_dims,
        hidden_layers,
        latent_dims,
    
    Returns:
        encoder,
        decoder,
        auto
    """

    # 1. Encoder
    encoder_input = keras.layers.Input(shape=(input_dims,))
    x = encoder_input

    for layer_dim in hidden_layers:
        x = keras.layers.Dense(layer_dim, activation='relu')(x)

    z_mean = keras.layers.Dense(latent_dims, name='z_mean')(x)
    z_log_var = keras.layers.Dense(latent_dims, name='z_log_var')(x)

    def sampling(args):
        z_mean, z_log_var = args
        epsilon = keras.backend.random_normal(
            shape=keras.backend.shape(z_mean),
            mean=0.0, stddev=1.0
        )
        return z_mean + keras.backend.exp(0.5 * z_log_var) * epsilon

    z = keras.layers.Lambda(sampling, output_shape=(latent_dims,))(
        [z_mean, z_log_var]
    )

    encoder = keras.models.Model(
        inputs=encoder_input, outputs=[z, z_mean, z_log_var],
    )

    # 2. Decoder
    decoder_input = keras.layers.Input(shape=(latent_dims,))
    x = decoder_input

    for layer_dim in reversed(hidden_layers):
        x = keras.layers.Dense(layer_dim, activation='relu')(x)

    decoder_output = keras.layers.Dense(
        input_dims, activation='sigmoid'
    )(x)

    decoder = keras.models.Model(
        inputs=decoder_input, outputs=decoder_output
    )

    # 3. VAE model
    auto_input = keras.layers.Input(shape=(input_dims,))
    z, z_mean, z_log_var = encoder(auto_input)
    auto_output = decoder(z)

    auto = keras.models.Model(
        inputs=auto_input, outputs=auto_output
    )

    kl_loss = -0.5 * keras.backend.sum(
        1 + z_log_var - keras.backend.square(z_mean)
        - keras.backend.exp(z_log_var),
        axis=-1
    )
    auto.add_loss(keras.backend.mean(kl_loss))

    auto.compile(
        optimizer=keras.optimizers.Adam(),
        loss=keras.losses.BinaryCrossentropy()
    )

    return encoder, decoder, auto
        
