#!/usr/bin/env python3
"""
Module: 4-train
"""

import tensorflow.keras as K


def train_model(
        network,
        data,
        labels, batch_size,
        epochs,
        validation_data=None,
        verbose=True, shuffle=False):
    """
    Function: train_model
    """

    History = network.fit(
        x=data,
        y=labels,
        batch_size=batch_size,
        epochs=epochs,
        verbose=verbose,
        shuffle=shuffle,
        validation_data=validation_data
    )

    return History
