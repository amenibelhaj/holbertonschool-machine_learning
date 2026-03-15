#!/usr/bin/env python3
"""Normalizes unactivated output of a neural network using batch normalization"""
import numpy as np


def batch_norm(Z, gamma, beta, epsilon):
    """
    Normalizes an unactivated output of a neural network 
    using batch normalization

    Args:
        Z: numpy.ndarray of shape (m, n) to be normalized
           m is the number of data points
           n is the number of features in Z
        gamma: numpy.ndarray of shape (1, n) containing the scales
        beta: numpy.ndarray of shape (1, n) containing the offsets
        epsilon: small number used to avoid division by zero

    Returns:
        The normalized Z matrix
    """
    # Calculate mean and variance along the batch axis (m)
    mean = np.mean(Z, axis=0)
    variance = np.var(Z, axis=0)

    # Standardize the data
    Z_centered = Z - mean
    Z_norm = Z_centered / np.sqrt(variance + epsilon)

    # Scale and shift using learnable parameters gamma and beta
    # Broadcasting handles the (1, n) shape against (m, n)
    return gamma * Z_norm + beta
