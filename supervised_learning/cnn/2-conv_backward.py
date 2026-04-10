#!/usr/bin/env python3
""" Module: 2-conv_backward """

import numpy as np


def conv_backward(dZ, A_prev, W, b, padding="same", stride=(1, 1)):
    """
    dZ is a numpy.ndarray of shape (m, h_new, w_new, c_new)
    containing the partial derivatives with respect to the unactivated output
    of the convolutional layer
    A_prev is a numpy.ndarray of shape (m, h_prev, w_prev, c_prev)
    containing the output of the previous layer
    W is a numpy.ndarray of shape (kh, kw, c_prev, c_new)
    containing the kernels for the convolution
    b is a numpy.ndarray of shape (1, 1, 1, c_new)
    containing the biases applied to the convolution
    padding is a string that is either same or valid,
    indicating the type of padding used
    stride is a tuple of (sh, sw) containing the strides for the convolution
        sh is the stride for the height
        sw is the stride for the width
    Returns: the partial derivatives with respect to
        the previous layer (dA_prev),
        the kernels (dW), and the biases (db), respectively
    """
    m, h_prev, w_prev, c_prev = A_prev.shape
    m, h_new, w_new, c_new = dZ.shape
    kh, kw, c_prev, c_new = W.shape
    sh, sw = stride

    if padding == 'same':
        ph = int(((h_prev - 1) * sh + kh - h_prev) / 2) + 1
        pw = int(((w_prev - 1) * sw + kw - w_prev) / 2) + 1
    else:
        ph = 0
        pw = 0

    dA_prev = np.zeros((m, h_prev, w_prev, c_prev))
    dW = np.zeros((kh, kw, c_prev, c_new))
    db = np.zeros((1, 1, 1, c_new))

    A_prev_pad = np.pad(A_prev, ((0, 0), (ph, ph),
                                 (pw, pw), (0, 0)), mode='constant')
    dA_prev_pad = np.pad(dA_prev, ((0, 0), (ph, ph),
                                   (pw, pw), (0, 0)), mode='constant')

    for i in range(m):
        a_prev_pad = A_prev_pad[i]
        da_prev_pad = dA_prev_pad[i]
        for h in range(h_new):
            for w in range(w_new):
                for c in range(c_new):
                    vert_start = h * sh
                    vert_end = vert_start + kh
                    horiz_start = w * sw
                    horiz_end = horiz_start + kw

                    a_slice = a_prev_pad[vert_start:vert_end,
                                         horiz_start:horiz_end, :]

                    dA_prev_pad[i, vert_start:vert_end,
                                horiz_start:horiz_end, :] +=\
                        W[:, :, :, c] * dZ[i, h, w, c]
                    dW[:, :, :, c] += a_slice * dZ[i, h, w, c]
                    db[:, :, :, c] += dZ[i, h, w, c]

    if padding == 'same':
        dA_prev = dA_prev_pad[:, ph:-ph, pw:-pw, :]
    else:
        dA_prev = dA_prev_pad

    return dA_prev, dW, db
