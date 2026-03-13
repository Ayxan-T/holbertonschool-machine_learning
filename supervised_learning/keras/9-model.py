#!/usr/bin/env python3
"""
save and load functions
"""

import tensorflow.keras as K


def save_model(network, filename):
    """
    Function: save_model
    """
    network.save(network, filename)
    return None

def load_model(filename):
    """
    Function: load_model
    """
    return K.models.load_model(filename)
