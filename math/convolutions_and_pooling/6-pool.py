#!/usr/bin/env python3
""" Module: 6-pool """
import numpy as np


def pool(images, kernel_shape, stride, mode='max'):
    """ Performs pooling on images """
    m, h, w, c = images.shape
    kh, kw = kernel_shape
    sh, sw = stride

    # Calculate output dimensions
    oh = (h - kh) // sh + 1
    ow = (w - kw) // sw + 1

    convolved = np.zeros((m, oh, ow, c))

    # Loop over output dimensions
    for i in range(oh):
        for j in range(ow):
            # Calculate start positions based on strides
            hs, ws = i * sh, j * sw
            # Extract slice and multiply by kernel
            receptive_field = images[:, hs:hs+kh, ws:ws+kw, :]
            
            if mode == 'max':
                convolved[:, i, j] = \
                    np.max(receptive_field, axis=(1,2))
            elif mode == 'avg':
                convolved[:, i, j] = \
                    np.average(receptive_field, axis=(1,2))

    return convolved
