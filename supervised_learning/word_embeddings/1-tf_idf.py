#!/usr/bin/env python3
"""This module provides a function for creating TF-IDF embedding matrix.

Function: tf_idf()
"""

from sklearn.feature_extraction.text import TfidfVectorizer


def tf_idf(sentences, vocab=None):
    """Creates TF-IDF embedding matrix from the given sentences.
    
    Args:
        sentences: (list) sentences to analyze
        vocab: (list) vocabulary words to use for the analysis
            if set to None, all words within sentences are used
    
    Returns:
        embeddings: (numpy.ndarray)[num_sentences, num_features]
        features: (list)
    """
    # TfidfVectorizer handles all parsing, lowercasing, symbol removal
    vectorizer = TfidfVectorizer(
        vocabulary=vocab,  # Use custom vocab if provided
        lowercase=True,
        token_pattern=r'[a-z]{2,}',  # Only alphabetic words
        stop_words=None  # Keep all words
    )
    
    embeddings = vectorizer.fit_transform(sentences).toarray()
    features = vectorizer.get_feature_names_out()
    
    return embeddings, features
