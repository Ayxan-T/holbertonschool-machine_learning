#!/usr/bin/env python3
"""Module: 2-rnn_decoder"""

import tensorflow as tf
SelfAttention = __import__('1-self_attention').SelfAttention


class RNNDecoder(tf.keras.layers.Layer):
    """Perform decoder part of RNN-based machine translation."""

    def __init__(self, vocab, embedding, units, batch):
        """Initialize RNNDecoder attributes."""
        super().__init__()
        self.embedding = tf.keras.layers.Embedding(vocab, embedding)
        self.gru = tf.keras.layers.GRU(
            units,
            return_sequences=True,
            return_state=True,
            recurrent_initializer="glorot_uniform"
        )
        self.F = tf.keras.layers.Dense(vocab)
        self.attention = SelfAttention(units)

    def call(self, x, s_prev, hidden_states):
        """Compute the next decoder output and hidden state."""
        # Calculate context vector: shape (batch, units)
        context, _ = self.attention(s_prev, hidden_states)

        # Embed input x: x shape (batch, 1) -> x embedding shape (batch, 1, embedding)
        x_embedded = self.embedding(x)

        # Expand context vector: shape (batch, 1, units)
        context = tf.expand_dims(context, 1)

        # Concatenate context vector with x (embedded) in that order
        # Result shape: (batch, 1, units + embedding)
        x_concat = tf.concat([context, x_embedded], axis=-1)

        # Pass concatenated tensor through GRU layer
        output, s = self.gru(x_concat, initial_state=s_prev)

        # Reshape/squeeze output from (batch, 1, units) to (batch, units)
        output = tf.reshape(output, (-1, output.shape[2]))

        # Pass through Dense layer to get vocabulary distribution
        y = self.F(output)

        return y, s
