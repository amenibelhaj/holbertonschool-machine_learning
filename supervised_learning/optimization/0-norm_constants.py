#!/usr/bin/env python3
import numpy as np


def normalization_constants(X):
    """
    Calculates the normalization (standardization) constants of a matrix.

    X is a numpy.ndarray of shape (m, nx) to normalize
        m is the number of data points
        nx is the number of features

    Returns: the mean and standard deviation of each feature, respectively
    """
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    return mean, std
