#!/usr/bin/env python3
"""
Module: 8-train
"""

import tensorflow.keras as K


def train_model(network, data,
                labels, batch_size,
                epochs, validation_data=None,
                early_stopping=False,
                patience=0, learning_rate_decay=False,
                alpha=0.1, decay_rate=1,
                save_best=False, filepath=None,
                verbose=True, shuffle=False):
    """
    Function: train_model
    """
    callbacks = []
    if early_stopping and validation_data is not None:

        early_stopping = K.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=patience
        )

        callbacks.append(early_stopping)

    if learning_rate_decay and validation_data is not None:

        # normally defined with epoch and lr parameters
        def reverse_time_decay(epoch):
            return alpha / (1 + decay_rate * epoch)

        lr_scheduler = K.callbacks \
                        .LearningRateScheduler(reverse_time_decay, verbose=1)

        callbacks.append(lr_scheduler)

    if save_best:

        best_saver = K.callbacks \
                      .ModelCheckpoint(filepath=filepath, save_only_best=True)

        callbacks.append(best_saver)

    History = network.fit(
        x=data,
        y=labels,
        batch_size=batch_size,
        epochs=epochs,
        verbose=verbose,
        shuffle=shuffle,
        validation_data=validation_data,
        callbacks=callbacks if callbacks else None
    )

    return History
