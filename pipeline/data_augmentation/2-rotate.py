#!/usr/bin/env python3
""" Module: 2-rotate """

import tensorflow as tf


def rotate_image(image):
    """
    Write a function def rotate_image(image):
        that rotates an image by 90 degrees counter-clockwise:
    image is a 3D tf.Tensor containing the image to rotate
    Returns the rotated image
    """
    rotated_image = tf.image.rot90(image)
    return rotated_image
