#!/usr/bin/env python3
"""
Updates weights and biases using Gradient Descent with L2 Regularization
"""
import numpy as np


def l2_reg_gradient_descent(Y, weights, cache, alpha, lambtha, L):
    """
    Updates neural network weights and biases in place

    Args:
        Y: one-hot numpy.ndarray of shape (classes, m) with correct labels
        weights: dictionary of weights and biases
        cache: dictionary of the outputs of each layer
        alpha: learning rate
        lambtha: L2 regularization parameter
        L: number of layers
    """
    m = Y.shape[1]
    # Initial dZ for the output layer (Softmax + Cross-Entropy)
    dz = cache['A' + str(L)] - Y

    for i in range(L, 0, -1):
        A_prev = cache['A' + str(i - 1)]
        W_key = 'W' + str(i)
        b_key = 'b' + str(i)
        
        # Calculate gradients
        # dW = (1/m) * (dz . A_prev.T) + (lambtha/m) * W
        dw = (np.matmul(dz, A_prev.T) / m) + (lambtha / m * weights[W_key])
        db = np.sum(dz, axis=1, keepdims=True) / m

        if i > 1:
            # Backpropagate dz to the previous layer
            # For tanh: g'(Z) = 1 - A^2
            W = weights[W_key]
            A_curr_prev = cache['A' + str(i - 1)]
            dz = np.matmul(W.T, dz) * (1 - (A_curr_prev ** 2))

        # Update weights and biases in place
        weights[W_key] = weights[W_key] - (alpha * dw)
        weights[b_key] = weights[b_key] - (alpha * db)
