#!/usr/bin/env python3
""" Module: 0-neural_style """

import numpy as np
import tensorflow as tf

class NST:
    style_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1',
                    'block4_conv1', 'block5_conv1']
    content_layer = 'block5_conv2'

    def __init__(self, style_image, content_image, alpha=1e4, beta=1):
        if not isinstance(style_image, np.ndarray) or \
        len(style_image.shape) != 3 or style_image.shape[2] != 3:
            raise TypeError("style_image must be a numpy.ndarray with shape"\
                            " (h, w, 3)")
        sh, sw, _ = style_image.shape

        if not isinstance(content_image, np.ndarray) or \
        len(content_image.shape) != 3 or content_image.shape[2] != 3:
            raise TypeError("content_image must be a numpy.ndarray with shape"\
                            " (h, w, 3)")
        ch, cw, _ = content_image.shape

        if not isinstance(alpha, (int, float)) or alpha < 0:
            raise TypeError("alpha must be a non-negative number")
        if not isinstance(beta, (int, float)) or beta < 0:
            raise TypeError("beta must be a non-negative number")

        self.style_image = self.scale_image(style_image)
        self.content_image = self.scale_image(content_image)
        self.alpha = alpha
        self.beta = beta
    
    @staticmethod
    def scale_image(image):
        if not isinstance(image, np.ndarray) or len(image.shape) != 3 \
            or image.shape[2] != 3:
            raise TypeError("image must be a numpy.ndarray with shape"\
                            " (h, w, 3)")

        new_image = tf.image.resize(
            image,
            size=[512, 512],
            preserve_aspect_ratio=True,
            method="bicubic"
        )

        new_image = new_image[tf.newaxis, ...]


        # preventing bicubic 'overshoot'
        return tf.clip_by_value(new_image, 0, 255) / 255.0
