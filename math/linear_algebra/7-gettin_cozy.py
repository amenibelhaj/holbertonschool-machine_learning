#!/usr/bin/env python3
"""
7-gettin_cozy
Module containing the function cat_matrices2D(mat1, mat2, axis=0)
that concatenates two matrices along a specific axis.
"""


def cat_matrices2D(mat1, mat2, axis=0):
    """
    Concatenates two 2D matrices along a specific axis.

    Args:
        mat1 (list of lists): The first 2D matrix.
        mat2 (list of lists): The second 2D matrix.
        axis (int): The axis along which to concatenate (0 for rows,
                    1 for columns).

    Returns:
        list of lists: A new matrix representing the concatenation,
                       or None if the matrices cannot be concatenated.
    """
    # ------------------
    # 1. Axis 0: Vertical Concatenation (Stacking)
    # ------------------
    if axis == 0:
        # Check Requirement: Number of columns must match.
        if len(mat1[0]) != len(mat2[0]):
            return None

        # If columns match, concatenate the rows (outer lists)
        # Using list slicing [:] ensures we return a new matrix
        new_matrix = [row[:] for row in mat1] + [row[:] for row in mat2]
        return new_matrix

    # ------------------
    # 2. Axis 1: Horizontal Concatenation (Side-by-side)
    # ------------------
    elif axis == 1:
        # Check Requirement: Number of rows (length of outer list) must match.
        if len(mat1) != len(mat2):
            return None

        # If rows match, we must concatenate corresponding ROWS element-wise.
        new_matrix = [
            # The continuation line starts 4 spaces plevel of new_matrix = [
            mat1[i][:] + mat2[i][:]
            for i in range(len(mat1))
        ]
        return new_matrix

    # ------------------
    # 3. Invalid Axis
    # ------------------
    else:
        return None
