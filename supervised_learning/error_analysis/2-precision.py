#!/usr/bin/env python3
"""
Module to calculate precision for each class in a confusion matrix
"""
import numpy as np


def precision(confusion):
    """
    Calculates the precision for each class in a confusion matrix

    Args:
        confusion: numpy.ndarray of shape (classes, classes)
                   row indices = correct labels, col indices = predicted

    Returns:
        precision: numpy.ndarray of shape (classes,)
                   containing the precision of each class
    """
    # True Positives are the diagonal elements
    tp = np.diag(confusion)

    # Total predicted observations for each class are the column sums
    # We use axis=0 to sum down the columns
    predicted_total = np.sum(confusion, axis=0)

    # Precision = TP / (TP + FP)
    return tp / predicted_total
