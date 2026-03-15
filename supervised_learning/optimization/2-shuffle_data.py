#!/usr/bin/env python3
"""Shuffles data points in two matrices"""
import numpy as np


def shuffle_data(X, Y):
    """
    Shuffles the data points in two matrices the same way
    
    Args:
        X: numpy.ndarray of shape (m, nx) to shuffle
        Y: numpy.ndarray of shape (m, ny) to shuffle
           m is the number of data points
           nx/ny are the number of features
           
    Returns:
        The shuffled X and Y matrices
    """
    # Create a list of indices from 0 to m-1
    # numpy.random.permutation returns a shuffled copy of this range
    m = X.shape[0]
    permutation = np.random.permutation(m)
    
    # Use the same shuffled index array to reorder both X and Y
    return X[permutation], Y[permutation]
