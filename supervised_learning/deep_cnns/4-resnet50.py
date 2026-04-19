#!/usr/bin/env python3
""" Module: 4-resnet50 """

from tensorflow import keras as K
# identity_block = __import__('2-identity_block').identity_block
# projection_block = __import__('3-projection_block').projection_block


def resnet50():
    """You can assume the input data will have shape (224, 224, 3)
    All convolutions inside and outside the blocks should be followed by batch normalization along the channels axis and a rectified linear activation (ReLU), respectively.
    All weights should use he normal initialization
    The seed for the he_normal initializer should be set to zero
    You may use:
    identity_block = __import__('2-identity_block').identity_block
    projection_block = __import__('3-projection_block').projection_block
    Returns: the keras model"""

    X_input = K.Input(shape=(224, 224, 3))
    he_normal = K.initializers.he_normal(seed=0)

    # Zero-Padding
    X = K.layers.ZeroPadding2D((3, 3))(X_input)

    # Stage 1
    X = K.layers.Conv2D(64, (7, 7), strides=(2, 2),
                        padding='same', kernel_initializer=he_normal)(X)
    X = K.layers.BatchNormalization(axis=3)(X)
    X = K.layers.Activation('relu')(X)
    X = K.layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same')(X)

    # Stage 2
    X = projection_block(X, filters=[64, 64, 256], s=1)
    X = identity_block(X, filters=[64, 64, 256])
    X = identity_block(X, filters=[64, 64, 256])

    # Stage 3
    X = projection_block(X, filters=[128, 128, 512], s=2)
    X = identity_block(X, filters=[128, 128, 512])
    X = identity_block(X, filters=[128, 128, 512])
    X = identity_block(X, filters=[128, 128, 512])

    # Stage 4
    X = projection_block(X, filters=[256, 256, 1024], s=2)
    X = identity_block(X, filters=[256, 256, 1024])
    X = identity_block(X, filters=[256, 256, 1024])
    X = identity_block(X, filters=[256, 256, 1024])
    X = identity_block(X, filters=[256, 256, 1024])
    X = identity_block(X, filters=[256, 256, 1024])

    # Stage 5
    X = projection_block(X, filters=[512, 512, 2048], s=2)
    X = identity_block(X, filters=[512, 512, 2048])
    X = identity_block(X, filters=[512, 512, 2048])

    # AVGPOOL
    X = K.layers.AveragePooling2D(pool_size=(2, 2), padding='same')(X)
    X = K.layers.Flatten()(X)
    X = K.layers.Dense(1000, activation='softmax', kernel_initializer=he_normal)(X)

    model = K.Model(inputs=X_input, outputs=X)

    return model