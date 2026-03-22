#!/usr/bin/env python3
"""
Module: ble
"""

import tensorflow as tf


def l2_reg_cost(cost, model):
    """
    Calculates the total cost for each layer of the network,
    accounting for L2 regularization.
    """
    # Sum all L2 penalty scalars into one scalar
    l2_total = model.losses

    # Add that scalar to the existing cost tensor
    return cost + l2_total
