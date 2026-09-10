#!/usr/bin/env python3
"""Module to build the ResNet-50 architecture."""
from tensorflow import keras as K

identity_block = __import__('2-identity_block').identity_block
projection_block = __import__('3-projection_block').projection_block


def resnet50():
    """Builds the ResNet-50 architecture as described in paper.

    Returns:
        the keras model
    """
    init = K.initializers.he_normal(seed=0)
    X_input = K.Input(shape=(224, 224, 3))

    # Stage 1: Conv1 & MaxPool
    X = K.layers.Conv2D(
        64, (7, 7), strides=(2, 2), padding='same', kernel_initializer=init
    )(X_input)
    X = K.layers.BatchNormalization(axis=3)(X)
    X = X = K.layers.ReLU()(X)
    X = K.layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same')(X)

    # Stage 2: conv2_x (3 blocks)
    X = projection_block(X, filters=[64, 64, 256], s=1)
    X = identity_block(X, filters=[64, 64, 256])
    X = identity_block(X, filters=[64, 64, 256])

    # Stage 3: conv3_x (4 blocks)
    X = projection_block(X, filters=[128, 128, 512], s=2)
    X = identity_block(X, filters=[128, 128, 512])
    X = identity_block(X, filters=[128, 128, 512])
    X = identity_block(X, filters=[128, 128, 512])

    # Stage 4: conv4_x (6 blocks)
    X = projection_block(X, filters=[256, 256, 1024], s=2)
    X = identity_block(X, filters=[256, 256, 1024])
    X = identity_block(X, filters=[256, 256, 1024])
    X = identity_block(X, filters=[256, 256, 1024])
    X = identity_block(X, filters=[256, 256, 1024])
    X = identity_block(X, filters=[256, 256, 1024])

    # Stage 5: conv5_x (3 blocks)
    X = projection_block(X, filters=[512, 512, 2048], s=2)
    X = identity_block(X, filters=[512, 512, 2048])
    X = identity_block(X, filters=[512, 512, 2048])

    # Classification Head
    X = K.layers.AveragePooling2D((7, 7))(X)
    X = K.layers.Dense(
        1000, activation='softmax', kernel_initializer=init
    )(X)

    return K.Model(inputs=X_input, outputs=X)