#!/usr/bin/env python3
""" Module: 2-neural_style """

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
        self.load_model()

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
        vgg = tf.keras.applications.VGG19(
            include_top=False,
            weights='imagenet'
        )

        vgg.trainable = False

        style_outputs = []
        content_output = None

        x = vgg.input
        for layer in vgg.layers[1:]:
            if isinstance(layer, tf.keras.layers.MaxPooling2D):
                x = tf.keras.layers.AveragePooling2D(
                    pool_size=layer.pool_size,
                    strides=layer.strides,
                    padding=layer.padding,
                    name=layer.name
                )(x)
            else:
                x = layer(x)

            if layer.name in self.style_layers:
                style_outputs.append(x)
            if layer.name == self.content_layer:
                content_output = x

        # style_layer_objects = [
        #   vgg.get_layer(name) for name in self.style_layers]
        # content_layer_object = vgg.get_layer(self.content_layer)

        # style_object_outputs = [obj.output for obj in style_layer_objects]
        # content_object_output = content_layer_object.output

        model_outputs = style_outputs + [content_output]

        self.model = tf.keras.Model(inputs=vgg.input, outputs=model_outputs)

        def gram_matrix(input_layer):
            """ Function: gram_matrix """
            is_valid_type = isinstance(input_layer, (tf.Variable, tf.Tensor))
            is_rank_4 = len(input_layer.shape) == 4
            if not is_valid_type or not is_rank_4:
                raise TypeError("input_layer must be a tensor of rank 4")

            map_count = input_layer.shape[-1]    # number of feature maps

            # initializing a tensor
            matrix = tf.zeros([1, map_count, map_count])

            # for i in range(map_count):
            #     for j in range(map_count):
            #         prods = input_layer[0, :, :, i] * input_layer[0, :, :, j]
            #         matrix[1, i, j] = tf.reduce_sum(prods)

            maps_flat = tf.reshape(input_layer, [-1, map_count])
            
            # MC by Values x Values by MC = MC by MC
            matrix = tf.matmul(maps_flat, maps_flat, transpose_a=True)

            # inplace dimension expansion
            return matrix[tf.newaxis, ...]

