#!/usr/bin/env python3
"""Defines a function that initializes variables for a Gaussian Mixture
Model."""
import numpy as np
kmeans = __import__('1-kmeans').kmeans


def initialize(X, k):
    """
    Initialize variables for a Gaussian Mixture Model.

    Args:
        X (numpy.ndarray): shape (n, d), the data set
        k (int): positive number of clusters

    Returns:
        tuple: (pi, m, S), or (None, None, None) on failure
            pi: numpy.ndarray of shape (k,), the priors for each cluster,
                initialized evenly
            m: numpy.ndarray of shape (k, d), the centroid means for each
                cluster, initialized with K-means
            S: numpy.ndarray of shape (k, d, d), the covariance matrices
                for each cluster, initialized as identity matrices
    """
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None, None, None
    if not isinstance(k, int) or k <= 0:
        return None, None, None

    d = X.shape[1]
    pi = np.full((k,), 1 / k)
    m, _ = kmeans(X, k)
    S = np.tile(np.identity(d), (k, 1, 1))

    return pi, m, S
