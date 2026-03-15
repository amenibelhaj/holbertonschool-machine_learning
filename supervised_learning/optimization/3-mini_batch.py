#!/usr/bin/env python3
"""Creates mini-batches for training"""
import numpy as np
shuffle_data = __import__('2-shuffle_data').shuffle_data


def create_mini_batches(X, Y, batch_size):
    """
    Creates mini-batches to be used for training a neural network

    Args:
        X: numpy.ndarray of shape (m, nx) representing input data
        Y: numpy.ndarray of shape (m, ny) representing the labels
        batch_size: number of data points in a batch

    Returns:
        list of mini-batches containing tuples (X_batch, Y_batch)
    """
    # 1. Shuffle the data so each batch is random
    X_shuffled, Y_shuffled = shuffle_data(X, Y)

    m = X.shape[0]
    mini_batches = []

    # 2. Iterate through the data in increments of batch_size
    for i in range(0, m, batch_size):
        # Slice the shuffled data from i to i + batch_size
        # Python slicing automatically handles the 'end' of the list
        # if i + batch_size > m, it just takes what is left.
        X_batch = X_shuffled[i:i + batch_size]
        Y_batch = Y_shuffled[i:i + batch_size]

        mini_batches.append((X_batch, Y_batch))

    return mini_batches
