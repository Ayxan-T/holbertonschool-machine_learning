#!/usr/bin/env python3
""" Module: 0-inception_block """

from tensorflow import keras as K


def inception_block(A_prev, filters):
    """ Performs inception block based on GoogLeNet architecture.

    Args:
        A_prev : the output from the previous layer
        filters : a tuple or list containing F1, F3R, F3, F5R, F5, FPP;
            F1 : the number of filters in the 1x1 convolution
            F3R : the number of filters in the 1x1 convolution before
                the 3x3 convolution
            F3 : the number of filters in the 3x3 convolution
            F5R : the number of filters in the 1x1 convolution before
                the 5x5 convolution
            F5 : the number of filters in the 5x5 convolution
            FPP : the number of filters in the 1x1 convolution after
                the max pooling

    Returns:
        A : the concatenated output of the inception block
    """

    F1, F3R, F3, F5R, F5, FPP = filters

    conv1 = K.layers.Conv2D(filters=F1, kernel_size=(1, 1), padding='same',
                            activation='relu')(A_prev)

    conv3_reduced = K.layers.Conv2D(filters=F3R, kernel_size=(1, 1),
                                    padding='same', activation='relu')(A_prev)
    conv3 = K.layers.Conv2D(filters=F3, kernel_size=(3, 3), padding='same',
                            activation='relu')(conv3_reduced)

    conv5_reduced = K.layers.Conv2D(filters=F5R, kernel_size=(1, 1),
                                    padding='same', activation='relu')(A_prev)
    conv5 = K.layers.Conv2D(filters=F5, kernel_size=(5, 5), padding='same',
                            activation='relu')(conv5_reduced)

    pool = K.layers.MaxPooling2D(pool_size=(3, 3), strides=(1, 1),
                                 padding='same')(A_prev)
    pool_conv = K.layers.Conv2D(filters=FPP, kernel_size=(1, 1),
                                padding='same', activation='relu')(pool)

    output = K.layers.Concatenate(axis=-1)([conv1, conv3, conv5, pool_conv])

    return output
