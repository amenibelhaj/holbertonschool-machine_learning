#!/usr/bin/env python3
"""
Module for forward propagation with Dropout
"""
import numpy as np


def dropout_forward_prop(X, weights, L, keep_prob):
    """
    Conducts forward propagation using Dropout

    Args:
        X: numpy.ndarray of shape (nx, m) containing input data
        weights: dictionary of weights and biases
        L: number of layers in the network
        keep_prob: probability that a node will be kept

    Returns:
        A dictionary containing the outputs of each layer and the dropout masks
    """
    cache = {}
    cache['A0'] = X

    for i in range(1, L + 1):
        W = weights['W' + str(i)]
        b = weights['b' + str(i)]
        A_prev = cache['A' + str(i - 1)]

        # Linear Sum
        Z = np.matmul(W, A_prev) + b

        if i == L:
            # Softmax for the last layer
            t = np.exp(Z)
            cache['A' + str(i)] = t / np.sum(t, axis=0, keepdims=True)
        else:
            # Tanh for hidden layers
            A = np.tanh(Z)

            # Create Dropout Mask
            # Random values < keep_prob become 1, else 0
            D = np.random.rand(A.shape[0], A.shape[1])
            D = (D < keep_prob).astype(int)

            # Apply mask and rescale (Inverted Dropout)
            A = (A * D) / keep_prob

            cache['D' + str(i)] = D
            cache['A' + str(i)] = A

    return cache
