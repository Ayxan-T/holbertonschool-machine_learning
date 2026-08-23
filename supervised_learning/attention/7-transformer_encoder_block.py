#!/usr/bin/env python3
"""Module: 7-transformer_encoder_block"""

import tensorflow as tf
MultiHeadAttention = __import__('6-multihead_attention').MultiHeadAttention


class EncoderBlock(tf.keras.layers.Layer):
    def __init__(self, dm, h, hidden, drop_rate=0.1):
        super().__init__()

        # Multi-Head Attention layer
        self.mha = MultiHeadAttention(dm, h)
        
        # Dense Feed-Forward layers
        self.dense_hidden = tf.keras.layers.Dense(hidden, activation='relu')
        self.dense_output = tf.keras.layers.Dense(dm)
        
        # Layer Normalization layers
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        
        # Dropout layers
        self.dropout1 = tf.keras.layers.Dropout(drop_rate)
        self.dropout2 = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, training, mask=None):
        """Custom call method.
        Args: x, training, mask=None
        Returns: """

        # Sub-layer 1: Multi-Head Attention
        attn_output, _ = self.mha(x, x, x, mask)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)  # Residual connection

        # Sub-layer 2: Feed Forward (dense_hidden -> dense_output)
        ffn_output = self.dense_hidden(out1)
        ffn_output = self.dense_output(ffn_output)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)  # Residual connection

        return out2
