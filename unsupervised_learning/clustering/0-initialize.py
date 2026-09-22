#!/usr/bin/env python3
"""Defines a function that initializes cluster centroids for K-means."""
import numpy as np


def initialize(X, k):
    """
    Initialize cluster centroids for K-means.

    Args:
        X (numpy.ndarray): shape (n, d), the dataset
            n: number of data points
            d: number of dimensions for each data point
        k (int): positive number of clusters

    Returns:
        numpy.ndarray: shape (k, d), the initialized centroids for each
            cluster, or None on failure
    """
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None
    if not isinstance(k, int) or k <= 0:
        return None

    return np.random.uniform(X.min(axis=0), X.max(axis=0),
                             size=(k, X.shape[1]))
