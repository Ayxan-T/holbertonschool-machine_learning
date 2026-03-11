#!/usr/bin/env python3
"""
Module: 6-train
"""

import tensorflow.keras as K


def train_model(network, data,
            labels, batch_size,
            epochs, validation_data=None,
            early_stopping=False,
            patience=0, verbose=True,
            shuffle=False):
    """
    Function: train_model
    """
    if early_stopping and validation_data is not None:

        early_stopping = K.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=patience
        )

        History = network.fit(
            x=data,
            y=labels,
            batch_size=batch_size,
            epochs=epochs,
            verbose=verbose,
            shuffle=shuffle,
            validation_data=validation_data,
            callbacks=[early_stopping]
        )

    else:
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
