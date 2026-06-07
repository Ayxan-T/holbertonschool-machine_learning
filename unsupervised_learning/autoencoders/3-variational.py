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
        x = keras.layer.Dense(layer_dim, activation='relu')(x)
    
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
        encoder_input, [z_mean, z_log_var, z])
    
    return encoder, None, None
        
