#!/usr/bin/env python3
"""
Module to calculate specificity for each class in a confusion matrix
"""
import numpy as np


def specificity(confusion):
    """
    Calculates the specificity for each class in a confusion matrix

    Args:
        confusion: numpy.ndarray of shape (classes, classes)
                   row indices = correct labels, col indices = predicted

    Returns:
        specificity: numpy.ndarray of shape (classes,)
                     containing the specificity of each class
    """
    # Total sum of all elements in the matrix
    total = np.sum(confusion)

    # True Positives: diagonal elements
    tp = np.diag(confusion)

    # Actual Positives (P): Row sums
    actual_pos = np.sum(confusion, axis=1)

    # Predicted Positives: Column sums
    pred_pos = np.sum(confusion, axis=0)

    # False Positives (FP): Predicted Positives - True Positives
    fp = pred_pos - tp

    # False Negatives (FN): Actual Positives - True Positives
    fn = actual_pos - tp

    # True Negatives (TN): Total - (TP + FP + FN)
    tn = total - (tp + fp + fn)

    # Specificity = TN / (TN + FP)
    return tn / (tn + fp)
