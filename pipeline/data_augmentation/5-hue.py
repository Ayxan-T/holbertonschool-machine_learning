#!/usr/bin/env python3
""" Module: 5-hue """

import tensorflow as tf


def change_hue(image, delta):
    """ Changes the hue of an image

    Args:
        image: A 3D tf.Tensor containing the image to change
        delta: The amount to change the hue by

    Returns:
        A tf.Tensor containing the hue-changed image
    """
    return tf.image.adjust_hue(image, delta)
