#!/usr/bin/env python3
""" This module provides a function, which initializes cluster centroids for
K-means.

Function: initialize()
"""

import numpy as np


def initialize(X, k):
    """ Initializes cluster centroids for K-means using multivariate uniform
    distribution.

    Args:
        X: a numpy.ndarray of shape (num_samples, num_dimensions) containing
            the dataset that will be used for clustering
        k: a positive integer containing the number of clusters

    Returns:
        centroids: a numpy.ndarray of shape (k, num_dimensions) containing
            the initialized centroids for each cluster, or None on failure
    """
    if k <= 0:
        return None

    try:
        lows = np.min(X, axis=0)
        highs = np.max(X, axis=0)
        d = X.shape[1]

        centroids = np.random.uniform(low=lows, high=highs, size=(k, d))
    except (np.core._exceptions.UFuncTypeError, IndexError, ValueError):
        centroids = None

    return centroids if centroids is not None else None
