#!/usr/bin/env python3
""" Module: 1-crop """

import tensorflow as tf


def crop_image(image, size):
    """ Crops an image using tf.image.crop_to_bounding_box

    Args:
        image: 3D tf.Tensor containing the image to crop
        size: tuple of (new_height, new_width) containing the size of the
              cropped image

    Returns:
        3D tf.Tensor containing the cropped image
    """
    new_height, new_width, _ = size
    original_height = tf.shape(image)[0]
    original_width = tf.shape(image)[1]

    offset_height = (original_height - new_height) // 2
    offset_width = (original_width - new_width) // 2

    cropped_image = tf.image.crop_to_bounding_box(
        image,
        offset_height=offset_height,
        offset_width=offset_width,
        target_height=new_height,
        target_width=new_width
    )

    return cropped_image