#!/usr/bin/env python3
"""
Module: 6-momentum.py
"""

import tensorflow as tf


def create_momentum_op(alpha, beta1):
    """
    Function: create_momentum_op
    """
    optimizer = tf.keras.optimizers.SGD(learning_rate=alpha, momentum=beta1)

    return optimizer
