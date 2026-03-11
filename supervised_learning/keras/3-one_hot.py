#!/usr/bin/env python3
"""
Module: 3-one_hot
"""

import tensorflow.keras as K


def one_hot(labels, classes=None):
    """
    Function: one_hot
    """

    one_hot = K.utils.to_categorical(labels, num_classes=classes)

    return one_hot
