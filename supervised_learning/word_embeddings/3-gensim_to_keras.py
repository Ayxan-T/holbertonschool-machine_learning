#!/usr/bin/env python3
"""This module provides a function for converting gensim Word2Vec to Keras Embedding.

Function: gensim_to_keras()
"""

from tensorflow.keras.layers import Embedding


def gensim_to_keras(model):
    """Converts a gensim word2vec model to a keras Embedding layer.
    
    Args:
        model: trained gensim word2vec model
    
    Returns:
        embedding: trainable keras Embedding layer
    """
    # Get vocabulary size and vector dimension
    vocab_size = len(model.wv)
    vector_size = model.wv.vector_size
    
    # Extract weights from gensim model
    weights = model.wv.vectors
    
    # Create Keras Embedding layer with gensim weights
    # input_dim = vocabulary size (including index 0 for unknown words)
    # output_dim = vector dimensions
    embedding = Embedding(
        input_dim=vocab_size,
        output_dim=vector_size,
        weights=[weights],
        trainable=True
    )
    
    return embedding
