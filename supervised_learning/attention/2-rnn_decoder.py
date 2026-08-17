#!/usr/bin/env python3
"""Module: 2-rnn_decoder"""

import tensorflow as tf
SelfAttention = __import__('1-self_attention').SelfAttention


class RNNDecoder(tf.keras.layers.Layer):
    """Perform decoder part of RNN-based machine translation."""

    def __init__(self, vocab, embedding, units, batch):
        super().__init__()
        self.embedding = tf.keras.layers.Embedding(vocab, embedding)
        self.attention = SelfAttention(units)
        self.gru = tf.keras.layers.GRU(
                units,
                return_state=True,
                recurrent_initializer="glorot_uniform"
            )
        self.F = tf.keras.layers.Dense(vocab)

    def call(self, x, s_prev, hidden_states):
        """Compute the next decoder output and hidden state."""
        context, _ = self.attention(s_prev, hidden_states)
        context = tf.expand_dims(context, axis=1)
        embedded = self.embedding(x)
        decoder_input = tf.concat([context, embedded], axis=-1)
        _, s = self.gru(decoder_input, initial_state=s_prev)
        y = self.F(s)
        return y, s


