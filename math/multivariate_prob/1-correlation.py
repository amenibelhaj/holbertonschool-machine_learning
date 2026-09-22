#!/usr/bin/env python3
"""Defines a function that calculates a correlation matrix."""
import numpy as np


def correlation(C):
    """
    Calculate a correlation matrix from a covariance matrix.

    Args:
        C (numpy.ndarray): shape (d, d), a covariance matrix
            d: number of dimensions

    Returns:
        numpy.ndarray: shape (d, d), the correlation matrix
    """
    if not isinstance(C, np.ndarray):
        raise TypeError("C must be a numpy.ndarray")
    if C.ndim != 2 or C.shape[0] != C.shape[1]:
        raise ValueError("C must be a 2D square matrix")

    std = np.sqrt(np.diag(C))

    return C / np.outer(std, std)
