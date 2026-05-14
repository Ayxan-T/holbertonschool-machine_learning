#!/usr/bin/env python3
""" Module: 0-pca """

import numpy as np

def pca(X, var=0.95):
    """ Performs PCA on the dataset. 
    
    Args:
        X: dataset of shape (num_datapoints, num_dimensions)
            (each row has a mean of 0)
        var: the fraction of the variance that the PCA transformation
            should maintain

    Returns: 
        w: the weights matrix of shape (num_dimensions, new_num_dimensions)
            that maintains var fraction of X's original variance
    """