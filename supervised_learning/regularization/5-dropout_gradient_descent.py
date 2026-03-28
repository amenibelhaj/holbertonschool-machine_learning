#!/usr/bin/env python3
"""
Updates weights with Dropout regularization using gradient descent
"""
import numpy as np


def dropout_gradient_descent(Y, weights, cache, alpha, keep_prob, L):
    """
    Updates neural network weights and biases in place with dropout

    Args:
        Y: one-hot numpy.ndarray of shape (classes, m) with correct labels
        weights: dictionary of weights and biases
        cache: dictionary of the outputs and dropout masks of each layer
        alpha: learning rate
        keep_prob: probability that a node will be kept
        L: number of layers
    """
    m = Y.shape[1]
    # Starting dz for the output layer (Softmax + Cross-Entropy)
    dz = cache['A' + str(L)] - Y

    for i in range(L, 0, -1):
        A_prev = cache['A' + str(i - 1)]
        W_key = 'W' + str(i)
        b_key = 'b' + str(i)
        W = weights[W_key]

        # Standard weight and bias gradients
        dw = np.matmul(dz, A_prev.T) / m
        db = np.sum(dz, axis=1, keepdims=True) / m

        if i > 1:
            # 1. Backpropagate dz to the previous layer
            # tanh derivative: g'(Z) = 1 - A^2
            A_curr_prev = cache['A' + str(i - 1)]
            dz = np.matmul(W.T, dz) * (1 - (A_curr_prev ** 2))

            # 2. Apply Dropout Mask to the gradient
            D_key = 'D' + str(i - 1)
            dz = (dz * cache[D_key]) / keep_prob

        # Update weights and biases in place
        weights[W_key] = weights[W_key] - (alpha * dw)
        weights[b_key] = weights[b_key] - (alpha * db)
