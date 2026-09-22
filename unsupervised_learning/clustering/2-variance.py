#!/usr/bin/env python3
"""Defines a function that calculates the total intra-cluster variance."""
import numpy as np


def variance(X, C):
    """
    Calculate the total intra-cluster variance for a data set.

    Args:
        X (numpy.ndarray): shape (n, d), the data set
        C (numpy.ndarray): shape (k, d), the centroid means for each
            cluster

    Returns:
        float: the total variance, or None on failure
    """
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None
    if (not isinstance(C, np.ndarray) or C.ndim != 2
            or C.shape[1] != X.shape[1]):
        return None

    distances = np.linalg.norm(X[:, np.newaxis] - C, axis=2)

    return np.sum(np.min(distances, axis=1) ** 2)
