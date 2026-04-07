#!/usr/bin/env python3
""" Module: 3-contrast """

import tensorflow as tf


def change_contrast(image, lower, upper):
    """ Randomly adjusts the contrast of an image.

    Args:
        image: A 3D tf.Tensor containing the image to change the contrast of.
        lower: A tf.Tensor containing the lower bound for random contrast factor range.
        upper: A tf.Tensor containing the upper bound for random contrast factor range.

    Returns:
        A tf.Tensor containing the contrast-changed image.
    """
    return tf.image.random_contrast(image, lower, upper)