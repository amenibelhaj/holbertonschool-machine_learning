#!/usr/bin/env python3
"""Defines a function that calculates the definiteness of a matrix."""
import numpy as np


def definiteness(matrix):
    """
    Calculate the definiteness of a matrix.

    Args:
        matrix (numpy.ndarray): shape (n, n), the matrix

    Returns:
        str: Positive definite, Positive semi-definite,
            Negative semi-definite, Negative definite or Indefinite,
            or None if matrix is not a valid symmetric matrix
    """
    if not isinstance(matrix, np.ndarray):
        raise TypeError("matrix must be a numpy.ndarray")

    if (matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]
            or matrix.size == 0 or not np.allclose(matrix, matrix.T)):
        return None

    eigenvalues = np.linalg.eigvalsh(matrix)
    eigenvalues[np.isclose(eigenvalues, 0)] = 0

    if np.all(eigenvalues > 0):
        return "Positive definite"
    if np.all(eigenvalues >= 0):
        return "Positive semi-definite"
    if np.all(eigenvalues < 0):
        return "Negative definite"
    if np.all(eigenvalues <= 0):
        return "Negative semi-definite"
    return "Indefinite"
