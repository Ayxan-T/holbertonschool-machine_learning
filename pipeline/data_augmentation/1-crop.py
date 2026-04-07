#!/usr/bin/env python3
""" Module: 1-crop """

import tensorflow as tf


def crop_image(image, size):
    """ Performs a random crop of an image

    Args:
        image: 3D tf.Tensor containing the image to crop
        size: tuple of (new_height, new_width) containing the size of the
              cropped image

    Returns:
        3D tf.Tensor containing the cropped image
    """
    cropped_image = tf.image.random_crop(
        image,
        size=size
    )

    return cropped_image