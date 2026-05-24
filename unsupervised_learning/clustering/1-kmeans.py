#!/usr/bin/env python3
""" This module provides a function that performs K-means clustering.

Function: kmeans()
"""

import numpy as np


def kmeans(X, k, iterations=1000):
    """ Performs K-means clustering. 
    
    Args:
        X: numpy.ndarray of shape (num_samples, num_dimensions)
            containing the dataset
        k: a positive integer containing the number of clusters
        iterations: a positive integer containing the maximum number of
            iterations that should be performed
    
    Returns:
        C: numpy.ndarray of shape (k, num_dimensions) containing the centroid
            means for each cluster
        clss: numpy.ndarray of shape (num_samples,) containing the index of
            the cluster in C that each data point belongs to
        OR
        (None, None) in failure
    """

    # Validate X
    if (type(X) is not np.ndarray
        or X.ndim != 2
        or (X.shape[0] == 0 or X.shape[1] == 0)):
        return None, None
    
    # Validate k
    if type(k) is not int or k <= 0:
        return None, None
    
    # Validate iterations
    if type(iterations) is not int or iterations <= 0:
        return None, None
    
    # Initialize centroids
    C = np.random.uniform(
        low=np.min(X, axis=0),
        high=np.max(X, axis=0),
        size=(k, X.shape[1])
    )

    # Initialize helper variable to hold centroids. It will be used
    # to compare new centroids to previous ones (early stopping)
    C_helper = np.zeros_like(C)

    # Initialize cluster labels (indices based on order of centroids)
    clss = np.zeros(X.shape[0])

    # Repeat the number of 'iterations'
    for _ in range(iterations):

        # Calculate distances from every data point to every centroid
        # C.shape: [k, d];  X.shape: [n, d] -> [n, 1, d] broadcasting;
        # distances.shape: [n, k]
        distances = np.sum((X[:, np.newaxis, :] - C) ** 2, axis=2)

        # Update cluster labels of all data points
        clss = np.argmin(distances, axis=1)

        # Update centroids to new centers of clusters   
        # Repeat every centroid
        for i in range(len(C)):

            # Select indices of elements belonging to that cluster
            elements = np.where(clss == i)

            # Reinitialize the centroid if it has no elements
            if len(elements) == 0:
                C_helper[i] = np.random.uniform(
                    low=np.min(X, axis=0), high=np.max(X, axis=0),
                    size=(C.shape[1])
                )

                # Skip the rest of iteration
                continue

            # Calculate the new location of centroid and update it
            C_helper[i] = np.mean(X[np.where(clss == i)], axis=0)

        # End the process if no change happened in centroid locations
        if np.allclose(C, C_helper):
            break

    return C, clss
