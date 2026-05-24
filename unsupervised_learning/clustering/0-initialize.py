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
    mins = np.min(X, axis=0)
    maxs = np.max(X, axis=0)
    d = X.shape[1]

    rng = np.random.default_rng()
    centroids = rng.integers(low=mins, high=maxs, size=(k, d))

    return centroids


