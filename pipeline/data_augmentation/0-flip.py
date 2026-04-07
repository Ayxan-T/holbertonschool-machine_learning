#!/usr/bin/env python3
""" Module: 0-flip """

import tensorflow as tf


def flip_image(image):
    """ Function: flip_image """
    return tf.image.flip_left_right(image)
