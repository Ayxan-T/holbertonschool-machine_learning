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
        or X.ndims != 2
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

    # Initialize class labels (indices based on order of centroids)
    clss = np.zeros(X.shape[0])

    # Repeat for the number of 'iterations'
    for _ in range(iterations):

        # Repeat for every data point
        for i, data in enumerate(X):

            # Calculate distance (Euclidean) to all cluster centroids
            distances = np.sum(np.squared(data - C), axis=1)

            # Update class label the closest cluster
            clss[i] = np.argmin(distances)
        
        # Repeat for every centroid
        for i in range(len(C)):

            # Calculate the new location of centroid
            new_loc = np.avg(X[np.where(clss == i)], axis=0)

            # Update the centroid
            C[i] = new_loc

    return C, clss

            # If a cluster contains no data points during the update step, reinitialize its centroid
            # If a cluster contains no data points during the update step, reinitialize its centroidIf no change in the cluster centroids occurs between iterations, your function should return
