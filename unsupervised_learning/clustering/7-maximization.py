#!/usr/bin/env python3
"""Defines a function that calculates the maximization step in the EM
algorithm for a GMM."""
import numpy as np


def maximization(X, g):
    """
    Calculate the maximization step in the EM algorithm for a GMM.

    Args:
        X (numpy.ndarray): shape (n, d), the data set
        g (numpy.ndarray): shape (k, n), the posterior probabilities for
            each data point in each cluster

    Returns:
        tuple: (pi, m, S), or (None, None, None) on failure
            pi: numpy.ndarray of shape (k,), the updated priors for each
                cluster
            m: numpy.ndarray of shape (k, d), the updated centroid means
                for each cluster
            S: numpy.ndarray of shape (k, d, d), the updated covariance
                matrices for each cluster
    """
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None, None, None
    n, d = X.shape
    if (not isinstance(g, np.ndarray) or g.ndim != 2
            or g.shape[1] != n):
        return None, None, None
    if not np.allclose(np.sum(g, axis=0), 1):
        return None, None, None

    k = g.shape[0]
    n_k = np.sum(g, axis=1)

    pi = n_k / n
    m = (g @ X) / n_k[:, np.newaxis]

    S = np.zeros((k, d, d))
    for i in range(k):
        diff = X - m[i]
        S[i] = (g[i] * diff.T) @ diff / n_k[i]

    return pi, m, S
