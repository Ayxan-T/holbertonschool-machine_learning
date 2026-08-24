#!/usr/bin/env python3
"""Module: 2-rnn_decoder"""

import tensorflow as tf

SelfAttention = __import__('1-self_attention').SelfAttention


class RNNDecoder(tf.keras.layers.Layer):
    """RNN Decoder class for machine translation using Bahdanau Attention."""

    def __init__(self, vocab, embedding, units, batch):
        """Initializes the RNNDecoder layer.

        Args:
            vocab (int): Size of the output vocabulary.
            embedding (int): Dimensionality of the embedding vector.
            units (int): Number of hidden units in the GRU cell.
            batch (int): Batch size.
        """
        super(RNNDecoder, self).__init__()

        self.embedding = tf.keras.layers.Embedding(
            input_dim=vocab,
            output_dim=embedding
        )
        self.gru = tf.keras.layers.GRU(
            units=units,
            return_sequences=True,
            return_state=True,
            recurrent_initializer='glorot_uniform'
        )
        self.F = tf.keras.layers.Dense(units=vocab)
        self.attention = SelfAttention(units)

    def call(self, x, s_prev, hidden_states):
        """Executes the forward pass of the decoder step.

        Args:
            x (Tensor): Tensor of shape (batch, 1) containing the previous target word.
            s_prev (Tensor): Tensor of shape (batch, units) containing previous decoder hidden state.
            hidden_states (Tensor): Tensor of shape (batch, input_seq_len, units) from encoder outputs.

        Returns:
            y (Tensor): Tensor of shape (batch, vocab) with predicted word distribution.
            s (Tensor): Tensor of shape (batch, units) with new decoder hidden state.
        """
        # Calculate context vector and attention weights using attention mechanism
        context, _ = self.attention(s_prev, hidden_states)

        # Convert previous target token into embedding vector: shape (batch, 1, embedding)
        x = self.embedding(x)

        # Concatenate context vector and embedded input along feature axis: shape (batch, 1, units + embedding)
        x = tf.concat([tf.expand_dims(context, 1), x], axis=-1)

        # Pass concatenated vector through GRU cell
        output, s = self.gru(x)

        # Reshape GRU output from (batch, 1, units) to (batch, units)
        output = tf.reshape(output, (-1, output.shape[2]))

        # Pass through final Dense layer to project to target vocabulary size
        y = self.F(output)

        return y, s