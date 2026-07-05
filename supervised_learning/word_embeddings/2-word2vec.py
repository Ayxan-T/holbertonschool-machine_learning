#!/usr/bin/env python3
"""This module provides a function for creating and training a Word2Vec model.

Function: word2vec_model()
"""

from gensim.models import Word2Vec


def word2vec_model(sentences, vector_size=100, min_count=5, window=5, negative=5, cbow=True, epochs=5, seed=0, workers=1):
    """Creates, builds and trains a gensim Word2Vec model.
    
    Args:
        sentences: (list) sentences to be trained on
        vector_size: (int) dimensionality of the embedding layer
        min_count: (int) minimum number of occurrences of a word
            for use in training
        window: (int) maximum distance between the current and predicted word
            within a sentence
        negative: (int) size of negative sampling
        cbow: (bool) training type; True for CBOW, False for Skip-gram
        epochs: (int) number of iterations to train over
        seed: (int) seed for the random number generator
        workers: (int) number of worker threads to train the model
    
    Returns:
        model: trained gensim Word2Vec model
    """
    # sg=0 for CBOW, sg=1 for Skip-gram
    sg = 0 if cbow else 1
    
    model = Word2Vec(
        sentences=sentences,
        vector_size=vector_size,
        min_count=min_count,
        window=window,
        negative=negative,
        sg=sg,
        epochs=epochs,
        seed=seed,
        workers=workers
    )
    
    return model
