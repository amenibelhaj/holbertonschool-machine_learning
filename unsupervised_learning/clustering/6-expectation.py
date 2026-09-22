#!/usr/bin/env python3
"""Defines a function that calculates the expectation step in the EM
algorithm for a GMM."""
import numpy as np
pdf = __import__('5-pdf').pdf


def expectation(X, pi, m, S):
    """
    Calculate the expectation step in the EM algorithm for a GMM.

    Args:
        X (numpy.ndarray): shape (n, d), the data set
        pi (numpy.ndarray): shape (k,), the priors for each cluster
        m (numpy.ndarray): shape (k, d), the centroid means for each
            cluster
        S (numpy.ndarray): shape (k, d, d), the covariance matrices for
            each cluster

    Returns:
        tuple: (g, l), or (None, None) on failure
            g: numpy.ndarray of shape (k, n), the posterior probabilities
                for each data point in each cluster
            l: the total log likelihood
    """
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None, None
    n, d = X.shape
    if not isinstance(pi, np.ndarray) or pi.ndim != 1:
        return None, None
    k = pi.shape[0]
    if not isinstance(m, np.ndarray) or m.shape != (k, d):
        return None, None
    if not isinstance(S, np.ndarray) or S.shape != (k, d, d):
        return None, None
    if not np.isclose(np.sum(pi), 1):
        return None, None

    weighted = np.zeros((k, n))
    for i in range(k):
        weighted[i] = pi[i] * pdf(X, m[i], S[i])

    total = np.sum(weighted, axis=0)
    g = weighted / total
    l = np.sum(np.log(total))

    return g, l
