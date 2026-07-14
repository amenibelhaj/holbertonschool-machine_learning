#!/usr/bin/env python3
"""
Pooling forward propagation
"""

import numpy as np


def pool_forward(A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """
    Performs forward propagation over a pooling layer

    A_prev: numpy.ndarray of shape (m, h_prev, w_prev, c_prev)
    kernel_shape: tuple (kh, kw)
    stride: tuple (sh, sw)
    mode: max or avg

    Returns:
        output of the pooling layer
    """

    m, h_prev, w_prev, c_prev = A_prev.shape

    kh, kw = kernel_shape
    sh, sw = stride

    # Calculate output dimensions
    h_new = int(((h_prev - kh) / sh) + 1)
    w_new = int(((w_prev - kw) / sw) + 1)

    # Initialize output
    A = np.zeros((m, h_new, w_new, c_prev))

    # Loop through examples
    for i in range(h_new):
        for j in range(w_new):

            # Define the window
            h_start = i * sh
            h_end = h_start + kh

            w_start = j * sw
            w_end = w_start + kw

            window = A_prev[:, h_start:h_end, w_start:w_end, :]

            if mode == 'max':
                A[:, i, j, :] = np.max(window, axis=(1, 2))

            elif mode == 'avg':
                A[:, i, j, :] = np.mean(window, axis=(1, 2))

    return A
