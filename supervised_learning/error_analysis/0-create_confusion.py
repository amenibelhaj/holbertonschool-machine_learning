#!/usr/bin/env python3
"""
Module to create a confusion matrix
"""
import numpy as np


def create_confusion_matrix(labels, logits):
    """
    Creates a confusion matrix from one-hot encoded labels and logits

    Args:
        labels: one-hot numpy.ndarray of shape (m, classes)
                containing the correct labels
        logits: one-hot numpy.ndarray of shape (m, classes)
                containing the predicted labels

    Returns:
        confusion: numpy.ndarray of shape (classes, classes)
                   with row indices as correct and columns as predicted
    """
    # Transpose labels to get (classes, m)
    # Dot product with logits (m, classes)
    # Resulting shape: (classes, classes)
    return np.matmul(labels.T, logits)
