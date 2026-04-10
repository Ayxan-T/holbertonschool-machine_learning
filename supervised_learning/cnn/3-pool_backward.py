#!/usr/bin/env python3
""" Module: 3-pool_backward """

import numpy as np


def pool_backward(dA, A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """Calculate gradients by performing back propagation over a pooling
    layer.

    Args:
        dA (numpy.ndarray): Global gradient of the output of the pooling layer,
            of shape (m, h_new, w_new, c_new).
        A_prev (numpy.ndarray): Previous layer's activations of shape 
            (m, h_prev, w_prev, c).
        kernel_shape (tuple): A tuple of (kh, kw) representing the height 
            and width of the kernel.
        stride (tuple): A tuple of (sh, sw) representing the vertical 
            and horizontal stride.
        mode (str): The pooling orientation, either 'max' or 'avg'.

    Returns:
        numpy.ndarray: The partial derivatives with respect to the 
            previous layer (dA_prev).
    """
    m, h_new, w_new, c = dA.shape
    m, h_prev, w_prev, c = A_prev.shape
    kh, kw = kernel_shape
    sh, sw = stride

    dA_prev = np.zeros(A_prev.shape)

    for i in range(m):
        for j in range(h_new):
            for k in range(w_new):
                h_start = j * sh
                h_end = h_start + kh
                w_start = k * sw
                w_end = w_start + kw

                if mode == 'avg':
                    dA_prev[i, h_start:h_end, w_start:w_end, :] += \
                        dA[i, j, k, :] / (kh * kw)
                    
                if mode == 'max':
                    argmax_row = np.argmax(
                        A_prev[m, h_start:h_end, w_start:w_end, :], axis=1)
                    argmax_col = np.argmax(
                        A_prev[m, h_start:h_end, w_start:w_end, :], axis=0)
                    
                    dA_prev[m, h_start + argmax_row,
                            w_start + argmax_col, :] += dA[m, j, k, :]    
    return dA_prev