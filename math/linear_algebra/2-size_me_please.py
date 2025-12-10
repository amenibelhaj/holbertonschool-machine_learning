#!/usr/bin/env python3
"""
Function to calculate the shape (dimensions) of a matrix.
"""


def matrix_shape(matrix):
    """
    Calculates the shape (dimensions) of a matrix.

    Args:
        matrix (list): The matrix whose shape is to be determined.
        Assumed to be uniform (all elements have the same shape).

    Returns:
        list: A list of integers representing the shape of the matrix.
    """
    shape = []
    current_level = matrix

    while isinstance(current_level, list):
        shape.append(len(current_level))

        if len(current_level) == 0:
            break
        else:
            current_level = current_level[0]

    return shape
