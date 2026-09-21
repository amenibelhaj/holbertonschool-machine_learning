#!/usr/bin/env python3
"""Defines a function that calculates the minor matrix of a matrix."""


def determinant(matrix):
    """
    Calculate the determinant of a square matrix.

    Args:
        matrix (list of lists): the square matrix

    Returns:
        the determinant of matrix
    """
    if matrix == [[]]:
        return 1
    n = len(matrix)
    if n == 1:
        return matrix[0][0]
    if n == 2:
        return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]

    det = 0
    for j in range(n):
        sub = [row[:j] + row[j + 1:] for row in matrix[1:]]
        det += ((-1) ** j) * matrix[0][j] * determinant(sub)
    return det


def minor(matrix):
    """
    Calculate the minor matrix of a matrix.

    Args:
        matrix (list of lists): the square matrix

    Returns:
        list of lists: the minor matrix of matrix
    """
    if (not isinstance(matrix, list) or len(matrix) == 0
            or not all(isinstance(row, list) for row in matrix)):
        raise TypeError("matrix must be a list of lists")

    n = len(matrix)
    if matrix == [[]] or not all(len(row) == n for row in matrix):
        raise ValueError("matrix must be a non-empty square matrix")

    if n == 1:
        return [[1]]

    minors = []
    for i in range(n):
        minor_row = []
        for j in range(n):
            sub = [row[:j] + row[j + 1:]
                   for k, row in enumerate(matrix) if k != i]
            minor_row.append(determinant(sub))
        minors.append(minor_row)

    return minors
