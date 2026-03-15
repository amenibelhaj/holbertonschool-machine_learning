#!/usr/bin/env python3
"""Standardizes a matrix"""


def normalize(X, m, s):
    """
    Normalizes (standardizes) a matrix X

    Args:
        X: numpy.ndarray of shape (d, nx) to normalize
           d is the number of data points
           nx is the number of features
        m: numpy.ndarray of shape (nx,) containing the mean of all features
        s: numpy.ndarray of shape (nx,) containing the stdev of all features

    Returns:
        The normalized X matrix
    """
    # X - m subtracts the mean from each feature
    # / s divides each feature by its standard deviation
    return (X - m) / s
