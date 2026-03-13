#!/usr/bin/env python3
"""
testing
"""

import tensorflow.keras as K


def test_model(network, data, labels, verbose=True):
    """
    Function: test_model
    """
    test_loss, test_acc = network.evaluate(
        x=data,
        y=labels,
        verbose=verbose
    )

    return [test_loss, test_acc]
