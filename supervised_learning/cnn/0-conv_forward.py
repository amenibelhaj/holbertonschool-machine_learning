#!/usr/bin/env python3
"""Convolutional forward propagation"""
import numpy as np


def conv_forward(A_prev, W, b, activation, padding="same", stride=(1, 1)):
    """
    Performs forward propagation over a convolutional layer

    Args:
        A_prev: numpy.ndarray of shape (m, h_prev, w_prev, c_prev)
        W: numpy.ndarray of shape (kh, kw, c_prev, c_new)
        b: numpy.ndarray of shape (1, 1, 1, c_new)
        activation: activation function
        padding: "same" or "valid"
        stride: tuple (sh, sw)

    Returns:
        The output of the convolutional layer
    """

    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw, _, c_new = W.shape
    sh, sw = stride

    # Calculate padding
    if padding == "same":
        ph = int(((h_prev - 1) * sh + kh - h_prev) / 2)
        pw = int(((w_prev - 1) * sw + kw - w_prev) / 2)
    elif padding == "valid":
        ph = 0
        pw = 0

    # Add padding to input
    A_pad = np.pad(
        A_prev,
        ((0, 0), (ph, ph), (pw, pw), (0, 0)),
        mode="constant"
    )

    # Output dimensions
    h_new = int(((h_prev + 2 * ph - kh) / sh) + 1)
    w_new = int(((w_prev + 2 * pw - kw) / sw) + 1)

    Z = np.zeros((m, h_new, w_new, c_new))

    # Convolution operation
    for i in range(h_new):
        for j in range(w_new):
            for k in range(c_new):
                vert_start = i * sh
                vert_end = vert_start + kh

                horiz_start = j * sw
                horiz_end = horiz_start + kw

                a_slice = A_pad[
                    :,
                    vert_start:vert_end,
                    horiz_start:horiz_end,
                    :
                ]

                Z[:, i, j, k] = np.sum(
                    a_slice * W[:, :, :, k],
                    axis=(1, 2, 3)
                )

    # Add bias and activation
    Z = Z + b
    A = activation(Z)

    return A
