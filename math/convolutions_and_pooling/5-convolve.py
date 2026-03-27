#!/usr/bin/env python3
""" Module: 5-convolve """
import numpy as np


def convolve(images, kernels, padding='same', stride=(1, 1)):
    """ Performs a convolution on images using multiple kernels """
    m, h, w, c = images.shape
    kh, kw, _, nk = kernels.shape
    sh, sw = stride

    if padding == 'same':
        # Calculate padding to ensure output is ceil(h/sh, w/sw)
        ph = int(((h - 1) * sh + kh - h) / 2) + 1 \
            if ((h - 1) * sh + kh - h) % 2 \
            else int(((h - 1) * sh + kh - h) / 2)
        pw = int(((w - 1) * sw + kw - w) / 2) + 1 \
            if ((w - 1) * sw + kw - w) % 2 \
            else int(((w - 1) * sw + kw - w) / 2)
    elif padding == 'valid':
        ph, pw = 0, 0
    else:
        ph, pw = padding

    # Apply padding once for all cases
    images_padded = np.pad(images, ((0, 0), (ph, ph), (pw, pw), (0, 0)),
                           mode='constant', constant_values=0)

    # Calculate output dimensions (standard formula)
    oh = (h + 2 * ph - kh) // sh + 1
    ow = (w + 2 * pw - kw) // sw + 1

    convolved = np.zeros((m, oh, ow, nk))

    # Loop over output dimensions
    for i in range(oh):
        for j in range(ow):
            # Calculate start positions in the padded image
            hs, ws = i * sh, j * sw
            # Extract slice and multiply by kernel
            receptive_field = images_padded[:, hs:hs+kh, ws:ws+kw, :]

            for k in range(nk):
                convolved[:, i, j, k] = \
                    np.sum(receptive_field *
                        kernels[:, :, :, k], axis=(1, 2, 3))

    return convolved
