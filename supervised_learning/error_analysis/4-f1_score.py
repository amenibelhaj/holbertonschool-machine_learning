#!/usr/bin/env python3
"""
Module to calculate the F1 score for each class in a confusion matrix
"""
import numpy as np
sensitivity = __import__('1-sensitivity').sensitivity
precision = __import__('2-precision').precision


def f1_score(confusion):
    """
    Calculates the F1 score for each class in a confusion matrix

    Args:
        confusion: numpy.ndarray of shape (classes, classes)
                   row indices = correct labels, col indices = predicted

    Returns:
        f1: numpy.ndarray of shape (classes,)
            containing the F1 score of each class
    """
    # Get the precision and sensitivity for all classes
    p = precision(confusion)
    s = sensitivity(confusion)

    # Calculate the harmonic mean
    # F1 = 2 * (precision * sensitivity) / (precision + sensitivity)
    f1 = 2 * (p * s) / (p + s)

    return f1
