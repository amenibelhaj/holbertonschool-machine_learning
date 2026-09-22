#!/usr/bin/env python3
"""Defines a function that calculates the probability density function
of a Gaussian distribution."""
import numpy as np


def pdf(X, m, S):
    """
    Calculate the probability density function of a Gaussian
    distribution.

    Args:
        X (numpy.ndarray): shape (n, d), the data points whose PDF should
            be evaluated
        m (numpy.ndarray): shape (d,), the mean of the distribution
        S (numpy.ndarray): shape (d, d), the covariance of the
            distribution

    Returns:
        numpy.ndarray: shape (n,), the PDF values for each data point,
            or None on failure
    """
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None
    d = X.shape[1]
    if not isinstance(m, np.ndarray) or m.shape != (d,):
        return None
    if not isinstance(S, np.ndarray) or S.shape != (d, d):
        return None

    diff = X - m
    inv = np.linalg.inv(S)
    det = np.linalg.det(S)

    exponent = -0.5 * np.sum((diff @ inv) * diff, axis=1)
    coefficient = 1 / np.sqrt(((2 * np.pi) ** d) * det)

    return np.maximum(coefficient * np.exp(exponent), 1e-300)
