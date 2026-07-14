#!/usr/bin/env python3
"""Convolutional back propagation"""
import numpy as np


def conv_backward(dZ, A_prev, W, b, padding="same", stride=(1, 1)):
    """
    Performs back propagation over a convolutional layer of a neural
    network

    dZ: numpy.ndarray of shape (m, h_new, w_new, c_new) containing the
        partial derivatives with respect to the unactivated output of the
        convolutional layer
    A_prev: numpy.ndarray of shape (m, h_prev, w_prev, c_prev) containing
        the output of the previous layer
    W: numpy.ndarray of shape (kh, kw, c_prev, c_new) containing the
        kernels for the convolution
    b: numpy.ndarray of shape (1, 1, 1, c_new) containing the biases
        applied to the convolution
    padding: string that is either 'same' or 'valid', indicating the type
        of padding used
    stride: tuple of (sh, sw) containing the strides for the convolution

    Returns: the partial derivatives with respect to the previous layer
        (dA_prev), the kernels (dW), and the biases (db), respectively
    """
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw, c_prev, c_new = W.shape
    m, h_new, w_new, c_new = dZ.shape
    sh, sw = stride

    if padding == "same":
        ph = int(((h_prev - 1) * sh + kh - h_prev) / 2) + 1
        pw = int(((w_prev - 1) * sw + kw - w_prev) / 2) + 1
    elif padding == "valid":
        ph, pw = 0, 0
    else:
        ph, pw = padding

    A_prev_pad = np.pad(
        A_prev,
        ((0, 0), (ph, ph), (pw, pw), (0, 0)),
        mode="constant",
        constant_values=0,
    )

    dA_prev_pad = np.zeros_like(A_prev_pad)
    dW = np.zeros_like(W)

    db = np.sum(dZ, axis=(0, 1, 2), keepdims=True)

    for i in range(h_new):
        for j in range(w_new):
            v_start = i * sh
            v_end = v_start + kh
            h_start = j * sw
            h_end = h_start + kw

            A_slice = A_prev_pad[:, v_start:v_end, h_start:h_end, :]

            for k in range(c_new):
                dZ_k = dZ[:, i, j, k].reshape(m, 1, 1, 1)

                dA_prev_pad[:, v_start:v_end, h_start:h_end, :] += (
                    W[:, :, :, k] * dZ_k
                )

                dW[:, :, :, k] += np.sum(
                    A_slice * dZ_k,
                    axis=0
                )

    if padding == "same":
        dA_prev = dA_prev_pad[
            :,
            ph:-ph if ph > 0 else None,
            pw:-pw if pw > 0 else None,
            :
        ]
    else:
        dA_prev = dA_prev_pad

    return dA_prev, dW, db
