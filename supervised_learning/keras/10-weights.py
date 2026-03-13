#!/usr/bin/env python3
"""
save and load weights
"""

import tensorflow.keras as K


def save_weights(network, filename, save_format='keras'):
    """
    Function: save_weights
    """
    network.save_weights(filename, save_format=save_format)
    return None

def load_weights(network, filename):
    """
    Function: load_weights
    """
    network.load_weights(filename)
    return None
