#!/usr/bin/env python3
"""
10-ill_use_my_scale
Module containing the function np_shape(matrix) that calculates
the shape of a numpy.ndarray.
"""


def np_shape(matrix):
    """
    Calculates the shape of a numpy.ndarray.

    Args:
        matrix (numpy.ndarray): The array whose shape is to be determined.

    Returns:
        tuple: A tuple of integers representing the shape of the array.
    """
    # Access the built-in .shape attribute of the NumPy array object.
    return matrix.shape
