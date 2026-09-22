#!/usr/bin/env python3
"""Defines a function that tests for the optimum number of clusters by
variance."""
import numpy as np
kmeans = __import__('1-kmeans').kmeans
variance = __import__('2-variance').variance


def optimum_k(X, kmin=1, kmax=None, iterations=1000):
    """
    Test for the optimum number of clusters by variance.

    Args:
        X (numpy.ndarray): shape (n, d), the data set
        kmin (int): minimum number of clusters to check for (inclusive)
        kmax (int): maximum number of clusters to check for (inclusive)
        iterations (int): maximum number of iterations for K-means

    Returns:
        tuple: (results, d_vars), or (None, None) on failure
            results: list of the outputs of K-means for each cluster size
            d_vars: list of the difference in variance from the smallest
                cluster size for each cluster size
    """
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None, None
    if kmax is None:
        kmax = X.shape[0]
    if not isinstance(kmin, int) or kmin <= 0:
        return None, None
    if not isinstance(kmax, int) or kmax <= 0 or kmin >= kmax:
        return None, None
    if not isinstance(iterations, int) or iterations <= 0:
        return None, None

    results = []
    variances = []
    for k in range(kmin, kmax + 1):
        C, clss = kmeans(X, k, iterations)
        results.append((C, clss))
        variances.append(variance(X, C))

    d_vars = [variances[0] - var for var in variances]

    return results, d_vars
