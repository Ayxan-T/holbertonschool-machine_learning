#!/usr/bin/env python3
""" Module: 1-neural_style """

import numpy as np
import tensorflow as tf


class NST:
    """ Class: NST (Neural Style Transfer) """
    style_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1',
                    'block4_conv1', 'block5_conv1']
    content_layer = 'block5_conv2'

    def __init__(self, style_image, content_image, alpha=1e4, beta=1):
        if not isinstance(style_image, np.ndarray) \
                or len(style_image.shape) != 3 or style_image.shape[2] != 3:
            raise TypeError("style_image must be a numpy.ndarray with shape"
                            " (h, w, 3)")
        sh, sw, _ = style_image.shape

        if not isinstance(content_image, np.ndarray) \
                or len(content_image.shape) != 3 \
                or content_image.shape[2] != 3:
            raise TypeError("content_image must be a numpy.ndarray with shape"
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
        self.model = self.load_model()

    @staticmethod
    def scale_image(image):
        if not isinstance(image, np.ndarray) or len(image.shape) != 3 \
                or image.shape[2] != 3:
            raise TypeError("image must be a numpy.ndarray with shape"
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
    
    def load_model(self):
      """ Function: load_model """
      vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')

      # vgg.trainable = False

      style_layer_objects = [vgg.get_layer(name) for name in self.style_layers]
      content_layer_object = vgg.get_layer(self.content_layer)

      style_object_outputs = [obj.output for obj in style_layer_objects]
      content_object_output = content_layer_object.output

      model_outputs = style_object_outputs + [content_object_output]

      return tf.keras.Model(inputs=vgg.input, outputs=model_outputs)
