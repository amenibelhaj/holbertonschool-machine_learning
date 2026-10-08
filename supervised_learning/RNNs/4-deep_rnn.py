#!/usr/bin/env python3
"""Forward propagation for a deep RNN"""
import numpy as np


def deep_rnn(rnn_cells, X, h_0):
    """
    Performs forward propagation for a deep RNN

    rnn_cells: list of RNNCell instances of length l (number of layers)
    X: numpy.ndarray of shape (t, m, i) with the data
    h_0: numpy.ndarray of shape (l, m, h) with the initial hidden states
    Returns: H, Y
        H: numpy.ndarray containing all of the hidden states
        Y: numpy.ndarray containing all of the outputs
    """
    t, m, _ = X.shape
    layers, _, h = h_0.shape
    o = rnn_cells[-1].Wy.shape[1]

    H = np.zeros((t + 1, layers, m, h))
    Y = np.zeros((t, m, o))
    H[0] = h_0

    for step in range(t):
        layer_input = X[step]
        for layer in range(layers):
            h_next, y = rnn_cells[layer].forward(H[step, layer], layer_input)
            H[step + 1, layer] = h_next
            layer_input = h_next
        Y[step] = y

    return H, Y
