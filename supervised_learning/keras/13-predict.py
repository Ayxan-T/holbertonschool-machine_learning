#!/usr/bin/env python3
"""
Module: 13-predict
"""

import tensorflow.keras as K


def predict(network, data, verbose=False):
    """
    Function: predict
    """
    return network.predict(data, verbose=verbose)
