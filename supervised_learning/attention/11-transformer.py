#!/usr/bin/env python3
"""Module: 11-transformer"""

import tensorflow as tf

Encoder = __import__('9-transformer_encoder').Encoder
Decoder = __import__('10-transformer_decoder').Decoder


class Transformer(tf.keras.Model):
    """Transformer network inheriting from tensorflow.keras.Model."""

    def __init__(self, N, dm, h, hidden, input_vocab, target_vocab,
                 max_seq_input, max_seq_target, drop_rate=0.1):
        """Initializes the Transformer model.

        Args:
            N (int): Number of blocks in the encoder and decoder.
            dm (int): Dimensionality of the model.
            h (int): Number of attention heads.
            hidden (int): Number of hidden units in the feed-forward network.
            input_vocab (int): Size of the input vocabulary.
            target_vocab (int): Size of the target vocabulary.
            max_seq_input (int): Maximum sequence length for the input.
            max_seq_target (int): Maximum sequence length for the target.
            drop_rate (float): Dropout probability rate.
        """
        super(Transformer, self).__init__()

        self.encoder = Encoder(
            N, dm, h, hidden, input_vocab, max_seq_input, drop_rate
        )
        self.decoder = Decoder(
            N, dm, h, hidden, target_vocab, max_seq_target, drop_rate
        )
        self.linear = tf.keras.layers.Dense(units=target_vocab)

    def call(self, inputs, target, training, encoder_mask, look_ahead_mask, decoder_mask):
        """Executes the forward pass of the Transformer network.

        Args:
            inputs (Tensor): Input tensor of shape (batch, input_seq_len).
            target (Tensor): Target tensor of shape (batch, target_seq_len).
            training (bool): Indicates whether the model is in training mode.
            encoder_mask (Tensor): Padding mask for the encoder.
            look_ahead_mask (Tensor): Look-ahead mask for the decoder.
            decoder_mask (Tensor): Padding mask for the decoder.

        Returns:
            Tensor: Output tensor of shape (batch, target_seq_len, target_vocab).
        """
        # Pass inputs through the encoder
        enc_output = self.encoder(inputs, training, encoder_mask)

        # Pass encoder output and targets through the decoder
        dec_output = self.decoder(
            target, enc_output, training, look_ahead_mask, decoder_mask
        )

        # Project output to target vocabulary size
        final_output = self.linear(dec_output)

        return final_output