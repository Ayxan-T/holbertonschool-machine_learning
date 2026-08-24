#!/usr/bin/env python3
"""Module: 10-transformer_decoder"""

import tensorflow as tf
import numpy as np

positional_encoding = __import__('4-positional_encoding').positional_encoding
DecoderBlock = __import__('8-transformer_decoder_block').DecoderBlock


class Decoder(tf.keras.layers.Layer):
    """Transformer Decoder class that inherits from tf.keras.layers.Layer."""

    def __init__(self, N, dm, h, hidden, target_vocab, max_seq_len,
                 drop_rate=0.1):
        """Initializes the Decoder layer.

        Args:
            N (int): Number of DecoderBlocks in the stack.
            dm (int): Dimensionality of the model.
            h (int): Number of attention heads.
            hidden (int): Number of hidden units in the feed-forward network.
            target_vocab (int): Size of the target vocabulary.
            max_seq_len (int): Maximum sequence length.
            drop_rate (float): Dropout probability rate.
        """
        super(Decoder, self).__init__()

        self.N = N
        self.dm = dm
        self.embedding = tf.keras.layers.Embedding(input_dim=target_vocab,
                                                   output_dim=dm)
        self.positional_encoding = positional_encoding(max_seq_len, dm)
        self.blocks = [
            DecoderBlock(dm, h, hidden, drop_rate)
            for _ in range(N)
        ]
        self.dropout = tf.keras.layers.Dropout(rate=drop_rate)

    def call(self, x, encoder_output, training, look_ahead_mask, padding_mask):
        """Executes the forward pass of the Decoder.

        Args:
            x (Tensor): Input tensor of shape (batch, target_seq_len).
            encoder_output (Tensor): Output from the encoder
            of shape (batch, input_seq_len, dm).
            training (bool): Indicates whether the model is in training mode.
            look_ahead_mask (Tensor): Mask for Masked MH attention layer.
            padding_mask (Tensor): Mask for Second MH attention layer.

        Returns:
            Tensor: Output tensor of shape (batch, target_seq_len, dm).
        """
        seq_len = tf.shape(x)[1]

        # Embed target tokens and scale by square root of model dimension
        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.dm, tf.float32))

        # Add positional encodings and apply dropout
        x += self.positional_encoding[:seq_len, :]
        x = self.dropout(x, training=training)

        # Pass through N decoder blocks sequentially
        for i in range(self.N):
            x = self.blocks[i](
                x,
                encoder_output,
                training,
                look_ahead_mask,
                padding_mask
            )

        return x
