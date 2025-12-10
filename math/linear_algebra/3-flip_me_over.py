#!/usr/bin/env python3
"""
3-flip_me_over
Module containing the function matrix_transpose(matrix)
that returns the transpose of a 2D matrix.
"""


def matrix_transpose(matrix):
    """
    Returns the transpose of a 2D matrix.

    Args:
        matrix (list of lists): The 2D matrix to be transposed.
                                Assumed to be non-empty and uniform.

    Returns:
    list of lists: A new matrix representing the transpose of the input matrix.
    """
    # 1. Get the original matrix dimensions (rows and columns)
    num_rows = len(matrix)

    # Check if the matrix is truly 2D.
    if num_rows == 0:
        # Should not happen based on problem constraints ("never empty"),
        # but prevents an error if the matrix is [[]]
        return []

    num_cols = len(matrix[0])

    # 2. Build the new transposed matrix by swapping rows (i) and columns (j)
    transposed_matrix = [
        # New Rows: Iterate over the old columns (j)
        [
            # New Columns: Iterate over the old rows (i)
            matrix[i][j]
            for i in range(num_rows)
        ]
        for j in range(num_cols)
    ]

    return transposed_matrix
