#!/usr/bin/env python3
"""
Module: 6-momentum.py
"""

import numpy as np
import tensorflow.keras as K


def create_momentum_op(alpha, beta1):
    """
    Function: create_momentum_op
    """
    optimizer = K.optimizers.SGD(learning_rate=alpha, momentum=beta1)

    return optimizer
