#!/usr/bin/env python3
"""Back propagation over a pooling layer"""

import numpy as np


def pool_backward(dA, A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """
    Performs back propagation over a pooling layer

    dA: numpy.ndarray of shape (m, h_new, w_new, c_new)
        containing partial derivatives with respect to the output
    A_prev: numpy.ndarray of shape (m, h_prev, w_prev, c)
        containing the previous layer output
    kernel_shape: tuple (kh, kw)
        kernel size
    stride: tuple (sh, sw)
        stride size
    mode: max or avg pooling

    Returns:
        dA_prev: partial derivatives with respect to A_prev
    """

    m, h_prev, w_prev, c = A_prev.shape
    kh, kw = kernel_shape
    sh, sw = stride

    dA_prev = np.zeros_like(A_prev)

    h_new = dA.shape[1]
    w_new = dA.shape[2]

    for i in range(h_new):
        for j in range(w_new):
            vert_start = i * sh
            vert_end = vert_start + kh

            horiz_start = j * sw
            horiz_end = horiz_start + kw

            A_slice = A_prev[:, vert_start:vert_end,
                             horiz_start:horiz_end, :]

            if mode == 'max':
                mask = np.equal(
                    A_slice,
                    np.max(A_slice, axis=(1, 2), keepdims=True)
                )

                dA_prev[:, vert_start:vert_end,
                        horiz_start:horiz_end, :] += (
                    mask * dA[:, i:i+1, j:j+1, :]
                )

            elif mode == 'avg':
                da = dA[:, i:i+1, j:j+1, :]
                average = da / (kh * kw)

                dA_prev[:, vert_start:vert_end,
                        horiz_start:horiz_end, :] += (
                    np.ones_like(A_slice) * average
                )

    return dA_prev
