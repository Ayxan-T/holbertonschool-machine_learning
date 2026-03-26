#!/usr/bin/env python3
"""
Module: 1-convolve_grayscale_same
"""

import numpy as np


def convolve_grayscale_same(images, kernel):
    """
    images is a numpy.ndarray with shape
    (m, h, w) containing multiple grayscale images
        m is the number of images
        h is the height in pixels of the images
        w is the width in pixels of the images
    kernel is a numpy.ndarray with shape
        (kh, kw) containing the kernel for the convolution
        kh is the height of the kernel
        kw is the width of the kernel
    Returns: a numpy.ndarray containing the convolved images

    SAME padding
    padding width for vertical axis
    h + 2p - kh + 1 = h
    2p = kh - 1
    p = (kh - 1) / 2
    """
    kh, kw = kernel.shape
    
    # Calculate pad width using formula in docstring
    ph = (kh - 1) / 2
    pw = (kw - 1) / 2

    images = np.pad(images, ((0,), (1,), (1,)), "constant", constant_values=(0))

    m, h, w = images.shape

    # Calculate output dimensions
    oh = h - kh + 1
    ow = w - kw + 1

    convolved = np.zeros((m, oh, ow))

    for i in range(oh):
        for j in range(ow):
            convolved[:, i, j] = np. \
                sum(images[:, i:i+kh, j:j+kw]*kernel, axis=(1, 2))

    return convolved
