#!/usr/bin/env python3
""" Module: 4-brightness """

import tensorflow as tf


def change_brightness(image, max_delta):
    """ Changes the brightness of an image.

    Args:
        image: A 3D tf.Tensor containing the image.
        max_delta: The maximum amount the brightness should be changed.

    Returns:
        A tf.Tensor containing the altered image.
    """
    return tf.image.random_brightness(image, max_delta)
