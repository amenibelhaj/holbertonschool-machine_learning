#!/usr/bin/env python3
"""Defines a function that performs K-means on a dataset."""
import numpy as np


def kmeans(X, k, iterations=1000):
    """
    Perform K-means on a dataset.

    Args:
        X (numpy.ndarray): shape (n, d), the dataset
            n: number of data points
            d: number of dimensions for each data point
        k (int): positive number of clusters
        iterations (int): positive maximum number of iterations

    Returns:
        tuple: (C, clss), or (None, None) on failure
            C: numpy.ndarray of shape (k, d), the centroid means for
                each cluster
            clss: numpy.ndarray of shape (n,), the index of the cluster
                in C that each data point belongs to
    """
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None, None
    if not isinstance(k, int) or k <= 0:
        return None, None
    if not isinstance(iterations, int) or iterations <= 0:
        return None, None

    low = X.min(axis=0)
    high = X.max(axis=0)
    C = np.random.uniform(low, high, size=(k, X.shape[1]))

    for _ in range(iterations):
        distances = np.linalg.norm(X[:, np.newaxis] - C, axis=2)
        clss = np.argmin(distances, axis=1)

        new_C = np.copy(C)
        for j in range(k):
            points = X[clss == j]
            if points.shape[0] == 0:
                new_C[j] = np.random.uniform(low, high)
            else:
                new_C[j] = points.mean(axis=0)

        if np.array_equal(new_C, C):
            return C, clss
        C = new_C

    distances = np.linalg.norm(X[:, np.newaxis] - C, axis=2)
    clss = np.argmin(distances, axis=1)

    return C, clss
