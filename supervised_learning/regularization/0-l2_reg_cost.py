#!/usr/bin/env python3
"""
Module to calculate the cost of a neural network with L2 regularization
"""
import numpy as np


def l2_reg_cost(cost, lambtha, weights, L, m):
    """
    Calculates the cost of a neural network with L2 regularization

    Args:
        cost: cost of the network without L2 regularization
        lambtha: regularization parameter
        weights: dictionary of the weights and biases of the neural network
        L: number of layers in the neural network
        m: number of data points used

    Returns:
        The cost of the network accounting for L2 regularization
    """
    l2_sum = 0

    # Loop through each layer to sum the squared weights
    for i in range(1, L + 1):
        key = 'W' + str(i)
        # Use np.linalg.norm for the Frobenius norm squared (sum of squares)
        l2_sum += np.linalg.norm(weights[key]) ** 2

    # Apply the L2 formula: cost + (lambda / (2 * m)) * sum(weights^2)
    l2_cost = cost + (lambtha / (2 * m)) * l2_sum

    return l2_cost
