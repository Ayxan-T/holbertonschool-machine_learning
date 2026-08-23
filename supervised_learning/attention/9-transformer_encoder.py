#!/usr/bin/env python3
"""Module: 9-transformer_encoder"""

import tensorflow as tf
positional_encoding = __import__('4-positional_encoding').positional_encoding
EncoderBlock = __import__('7-transformer_encoder_block').EncoderBlock


class Encoder(tf.keras.layers.Layer):
    """Transformer Encoder"""
    def __init__(
            self, N, dm, h, hidden, input_vocab, max_seq_len, drop_rate=0.1):
        super().__init__()

        self.N = N
        self.dm = dm
        self.embedding = tf.keras.layers.Embedding(input_vocab, dm)
        self.positional_encoding = positional_encoding(max_seq_len, dm)
        self.blocks = [EncoderBlock(dm, h, hidden, drop_rate)
                       for _ in range(N)]
        self.dropout = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, training, mask):
        """Performs encoder mechanism.
        Args: x (batch, seq_len), training, mask"""
        seq_len = tf.shape(x)[1]

        x = self.embedding(x)  # (batch, seq_len, dm)
        x *= tf.math.sqrt(tf.cast(self.dm, tf.float32))

        # 1. Add positional encoding up to input_seq_len
        # Cast to x.dtype in case x is float32/float64 and positional_encoding
        # is float64/float32
        pos_encoding = tf.cast(
            self.positional_encoding[:seq_len, :], dtype=x.dtype)
        x += pos_encoding

        # 2. Apply dropout to the sum of input and positional encoding
        x = self.dropout(x, training=training)

        # 3. Pass sequentially through each EncoderBlock
        for block in self.blocks:
            x = block(x, training, mask)

        return x
