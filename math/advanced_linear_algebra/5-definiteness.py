#!/usr/bin/env python3
""" Module: 5-definiteness """

import numpy as np


def definiteness(matrix):
    """ Function: definiteness """
    # Step 1: Check if the input is a numpy array
    if not isinstance(matrix, np.ndarray):
        raise TypeError("matrix must be a numpy.ndarray")

    # Step 2: Check if it's a valid square, symmetric matrix
    # Definiteness is standardly applied to symmetric matrices
    if len(matrix.shape) != 2 or matrix.shape[0] != matrix.shape[1]:
        return None

    if not np.allclose(matrix, matrix.T):
        return None

    try:
        # Step 3: Calculate eigenvalues
        eigenvalues = np.linalg.eigvals(matrix)

        # Step 4: Evaluate the signs of the eigenvalues
        pos = np.all(eigenvalues > 0)
        pos_semi = np.all(eigenvalues >= 0)
        neg = np.all(eigenvalues < 0)
        neg_semi = np.all(eigenvalues <= 0)

        # Step 5: Return the correct classification
        if pos:
            return "Positive definite"
        elif pos_semi:
            return "Positive semi-definite"
        elif neg:
            return "Negative definite"
        elif neg_semi:
            return "Negative semi-definite"

        # If there are both positive and negative values
        if any(eigenvalues > 0) and any(eigenvalues < 0):
            return "Indefinite"

        return None

    except Exception:
        # Catch-all for any calculation errors
        return None
