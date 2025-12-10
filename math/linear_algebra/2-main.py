#!/usr/bin/env python3
"""
2-size_me_please
Module containing the function matrix_shape(matrix)
that calculates the shape (dimensions) of a nested list matrix.
"""

def matrix_shape(matrix):
    """
    Calculates the shape (dimensions) of a matrix represented as a nested list.
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