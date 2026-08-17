#!/usr/bin/env python3
"""Module: 0-rnn_encoder"""

import tensorflow as tf


class RNNEncoder(tf.keras.layers.Layer):
    """Encode a sequence using a GRU recurrent layer."""

    def __init__(self, vocab, embedding, units, batch):
        super().__init__()
        self.batch = batch
        self.units = units

        # lookup table for embeddings
        self.embedding = tf.keras.layers.Embedding(vocab, embedding)
        self.gru = tf.keras.layers.GRU(
            units,
            return_sequences=True,
            return_state=True,
            recurrent_initializer="glorot_uniform"
        )

    def initialize_hidden_state(self):
        """Initializes the hidden states for the RNN cell as tensor of zeros"""
        return tf.zeros(shape=(self.batch, self.units))

    def call(self, x, initial):
        """Return the GRU outputs and the final hidden state."""
        embedded = self.embedding(x)
        outputs, hidden = self.gru(embedded)
        return outputs, hidden
