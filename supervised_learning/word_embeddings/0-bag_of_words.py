#!/usr/bin/env python3
"""This module provides a function for creating Bag of Words embedding matrix.

Function: bag_of_words()
"""

from sklearn.feature_extraction.text import CountVectorizer


def bag_of_words(sentences, vocab=None):
    """Creates BoW embedding matrix from the given sentences.
    
    Args:
        sentences: (list) sentences to analyze
        vocab: (list) vocabulary words to use for the analysis
            if set to None, all words within sentences are used
    
    Returns:
        embeddings: (numpy.ndarray)[num_sentences, num_features]
        features: (list)
    """
    # CountVectorizer handles all parsing, lowercasing, symbol removal automatically
    vectorizer = CountVectorizer(
        vocabulary=vocab,  # Use custom vocab if provided
        lowercase=True,
        token_pattern=r'[a-z]+',  # Only alphabetic words (filters symbols)
        stop_words=None  # Keep all words
    )
    
    embeddings = vectorizer.fit_transform(sentences).toarray()
    features = vectorizer.get_feature_names_out().tolist()
    
    return embeddings, features
