#!/usr/bin/env python3
"""
Module: 2-convolve_grayscale_padding
"""

import numpy as np


def convolve_grayscale_padding(images, kernel, padding):
    """
    images is a numpy.ndarray with shape (m, h, w) containing multiple grayscale images
        m is the number of images
        h is the height in pixels of the images
        w is the width in pixels of the images
    kernel is a numpy.ndarray with shape (kh, kw) containing the kernel for the convolution
        kh is the height of the kernel
        kw is the width of the kernel
    padding is a tuple of (ph, pw)
        ph is the padding for the height of the image
        pw is the padding for the width of the image
    the image is padded with 0s
    Returns: a numpy.ndarray containing the convolved images
    """
    kh, kw = kernel.shape
    m, h, w = images.shape
    ph, pw = padding

    padded_images = np.pad(images,
                           ((0,), (ph,), (pw,)), "constant", constant_values=0)

    oh = h + 2*ph - kh + 1
    ow = w + 2*pw - kw + 1

    convolved = np.zeros((m, oh, ow))

    for i in range(h):
        for j in range(w):
            print(f"loop {i}-{j}")
            convolved[:, i, j] = np. \
                sum(padded_images[:, i:i+kh, j:j+kw]*kernel, axis=(1, 2))

    return convolved
