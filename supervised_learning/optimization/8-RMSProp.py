#!/usr/bin/env python3
"""
Module: 8-RMSProp
"""

import tensorflow as tf


def create_RMSProp_op(alpha, beta2, epsilon):
    """
    Function: create_RMSProp_op
    """
    return tf.keras.optimizers.RMSProp(learning_rate=alpha, rho=beta2, epsilon=epsilon)
