#!/usr/bin/env python3
"""
5-across_the_planes
Module containing the function add_matrices2D(mat1, mat2)
that adds two 2D matrices element-wise.
"""


# Note the extra blank line above this function definition to fix E302
def add_matrices2D(mat1, mat2):
    """
    Adds two 2D matrices element-wise.

    Args:
        mat1 (list of lists): The first 2D matrix.
        mat2 (list of lists): The second 2D matrix.

    Returns:
        list of lists: A new matrix containing the element-wise sums,
                       or None if the matrices are not the same shape.
    """
    # 1. Check Row Dimension (Number of rows, m)
    if len(mat1) != len(mat2):
        return None

    # Handle empty matrix case
    if len(mat1) == 0:
        return []

    # 2. Check Column Dimension (Number of columns, n)
    # Get the expected number of columns from the first row of mat1
    num_cols = len(mat1[0])

    # Check that mat1's rows are uniform AND mat2's rows match that length
    for row in mat1:
        if len(row) != num_cols:
            return None

    for row in mat2:
        if len(row) != num_cols:
            return None

    # 3. Perform Element-wise Addition
    # This block
    new_matrix = [
        # Outer loop: Iterates through rows (i)
        [
            # Inner loop: Iterates through columns (j)
            mat1[i][j] + mat2[i][j]
            for j in range(num_cols)
        ]
        for i in range(len(mat1))
    ]

    return new_matrix
