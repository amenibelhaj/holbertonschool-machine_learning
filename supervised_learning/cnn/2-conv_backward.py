#!/usr/bin/env python3

import numpy as np


def conv_backward(dZ, A_prev, W, b, padding="same", stride=(1, 1)):
    """
    Performs back propagation over a convolutional layer of a neural network

    dZ: numpy.ndarray of shape (m, h_new, w_new, c_new)
        gradient of the cost with respect to the output of the convolutional layer

    A_prev: numpy.ndarray of shape (m, h_prev, w_prev, c_prev)
        output of the previous layer

    W: numpy.ndarray of shape (kh, kw, c_prev, c_new)
        weights of the convolution

    b: numpy.ndarray of shape (1, 1, 1, c_new)
        biases of the convolution

    padding: "same" or "valid"

    stride: tuple containing the strides for height and width

    Returns:
        dA_prev, dW, db
    """

    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw, _, c_new = W.shape
    sh, sw = stride

    # Padding calculation
    if padding == "same":
        ph = int(((h_prev - 1) * sh + kh - h_prev) / 2)
        pw = int(((w_prev - 1) * sw + kw - w_prev) / 2)
    else:
        ph = pw = 0

    # Pad previous activation
    A_pad = np.pad(
        A_prev,
        ((0, 0), (ph, ph), (pw, pw), (0, 0)),
        mode='constant'
    )

    dA_pad = np.zeros_like(A_pad)
    dW = np.zeros_like(W)
    db = np.sum(dZ, axis=(0, 1, 2), keepdims=True)

    h_new = dZ.shape[1]
    w_new = dZ.shape[2]

    for i in range(m):
        for h in range(h_new):
            for w in range(w_new):
                for c in range(c_new):

                    vert_start = h * sh
                    vert_end = vert_start + kh

                    horiz_start = w * sw
                    horiz_end = horiz_start + kw

                    a_slice = A_pad[
                        i,
                        vert_start:vert_end,
                        horiz_start:horiz_end,
                        :
                    ]

                    dA_pad[
                        i,
                        vert_start:vert_end,
                        horiz_start:horiz_end,
                        :
                    ] += W[:, :, :, c] * dZ[i, h, w, c]

                    dW[:, :, :, c] += a_slice * dZ[i, h, w, c]

    if padding == "same":
        dA_prev = dA_pad[:, ph:-ph, pw:-pw, :]
    else:
        dA_prev = dA_pad

    return dA_prev, dW, db