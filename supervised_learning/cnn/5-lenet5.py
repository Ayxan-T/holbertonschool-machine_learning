#!/usr/bin/env python3
"""Module that contains the lenet5 function"""
from tensorflow import keras as K


def lenet5(X):
    """
    Builds a modified version of the LeNet-5 architecture using Keras.

    Parameters:
        X: K.Input of shape (m, 28, 28, 1) containing the input images

    Returns:
        A K.Model compiled to use Adam optimizer and accuracy metrics
    """
    initializer = K.initializers.he_normal(seed=0)

    # 1. Convolutional layer: 6 kernels, 5x5, same padding, ReLU
    conv1 = K.layers.Conv2D(
        filters=6,
        kernel_size=(5, 5),
        padding='same',
        activation='relu',
        kernel_initializer=initializer
    )(X)

    # 2. Max pooling layer: 2x2 kernel, 2x2 stride
    pool1 = K.layers.MaxPooling2D(
        pool_size=(2, 2),
        strides=(2, 2)
    )(conv1)

    # 3. Convolutional layer: 16 kernels, 5x5, valid padding, ReLU
    conv2 = K.layers.Conv2D(
        filters=16,
        kernel_size=(5, 5),
        padding='valid',
        activation='relu',
        kernel_initializer=initializer
    )(pool1)

    # 4. Max pooling layer: 2x2 kernel, 2x2 stride
    pool2 = K.layers.MaxPooling2D(
        pool_size=(2, 2),
        strides=(2, 2)
    )(conv2)

    # Flatten the feature maps for the fully connected layers
    flatten = K.layers.Flatten()(pool2)

    # 5. Fully connected layer: 120 nodes, ReLU
    fc1 = K.layers.Dense(
        units=120,
        activation='relu',
        kernel_initializer=initializer
    )(flatten)

    # 6. Fully connected layer: 84 nodes, ReLU
    fc2 = K.layers.Dense(
        units=84,
        activation='relu',
        kernel_initializer=initializer
    )(fc1)

    # 7. Fully connected softmax output layer: 10 nodes
    output = K.layers.Dense(
        units=10,
        activation='softmax',
        kernel_initializer=initializer
    )(fc2)

    # Construct the model
    model = K.Model(inputs=X, outputs=output)

    # Compile the model with Adam optimizer and accuracy metrics
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model
