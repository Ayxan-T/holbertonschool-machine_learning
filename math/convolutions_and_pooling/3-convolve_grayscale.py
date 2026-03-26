#!/usr/bin/env python3
"""
Module: 3-convolve_grayscale
"""

import numpy as np


def convolve_grayscale(images, kernel, padding='same', stride=(1, 1)):
    """
    images is a numpy.ndarray with shape (m, h, w) containing multiple grayscale images
        m is the number of images
        h is the height in pixels of the images
        w is the width in pixels of the images
    kernel is a numpy.ndarray with shape (kh, kw) containing the kernel for the convolution
        kh is the height of the kernel
        kw is the width of the kernel
    padding is either a tuple of (ph, pw), 'same', or 'valid'
        if 'same', performs a same convolution
        if 'valid', performs a valid convolution
        if a tuple:
            ph is the padding for the height of the image
            pw is the padding for the width of the image
    the image is padded with 0s
    stride is a tuple of (sh, sw)
        sh is the stride for the height of the image
        sw is the stride for the width of the image
    Returns: a numpy.ndarray containing the convolved images
    """
    kh, kw = kernel.shape
    m, h, w = images.shape
    
    if padding == "same":
            ph = kh // 2
            pw = kw // 2
            images = np.pad(images,
                           ((0,), (ph,), (pw,)), "constant", constant_values=0)
            oh, ow = h, w
    elif padding == "valid":
            oh = h - kh + 1
            ow = w - kw + 1
    else:
            ph, pw = padding
            images = np.pad(images,
                           ((0,), (ph,), (pw,)), "constant", constant_values=0)
            oh = h + 2*ph - kh + 1
            ow = w + 2*pw - kw + 1
    
    convolved = np.zeros((m, oh, ow))

    for i in range(oh):
        for j in range(ow):
            convolved[:, i, j] = np. \
                sum(images[:, i:i+kh, j:j+kw]*kernel, axis=(1, 2))

    return convolved
