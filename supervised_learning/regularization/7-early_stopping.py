#!/usr/bin/env python3
"""Module containing the early_stopping function."""


def early_stopping(cost, opt_cost, threshold, patience, count):
    """Determines if gradient descent should stop early.

    Args:
        cost: current validation cost of the neural network
        opt_cost: lowest recorded validation cost of the neural network
        threshold: threshold used for early stopping
        patience: patience count used for early stopping
        count: count of how long the threshold has not been met

    Returns:
        tuple: (boolean indicating whether to stop early, updated count)
    """
    if opt_cost - cost > threshold:
        count = 0
    else:
        count += 1

    return count >= patience, count