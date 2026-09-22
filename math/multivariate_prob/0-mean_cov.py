#!/usr/bin/env python3
"""Defines a function that calculates the mean and covariance of a data
set."""
import numpy as np


def mean_cov(X):
    """
    Calculate the mean and covariance of a data set.

    Args:
        X (numpy.ndarray): shape (n, d), the data set
            n: number of data points
            d: number of dimensions in each data point

    Returns:
        tuple: (mean, cov)
            mean: numpy.ndarray of shape (1, d), the mean of the data set
            cov: numpy.ndarray of shape (d, d), the covariance matrix
    """
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        raise TypeError("X must be a 2D numpy.ndarray")

    n = X.shape[0]
    if n < 2:
        raise ValueError("X must contain multiple data points")

    mean = np.mean(X, axis=0, keepdims=True)
    centered = X - mean
    cov = (centered.T @ centered) / (n - 1)

    return mean, cov
