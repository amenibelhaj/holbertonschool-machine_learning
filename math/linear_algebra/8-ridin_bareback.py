#!/usr/bin/env python3
"""
8-ridin_bareback
Module containing the function mat_mul(mat1, mat2)
that performs matrix multiplication on two 2D matrices.
"""


def mat_mul(mat1, mat2):
    """
    Performs matrix multiplication on two 2D matrices.

    Args:
        mat1 (list of lists): The first 2D matrix (A).
        mat2 (list of lists): The second 2D matrix (B).

    Returns:
        list of lists: A new matrix (C) representing the product A * B,
                       or None if the matrices cannot be multiplied.
    """
    # Dimensions of A: m x n
    m = len(mat1)
    if m == 0 or len(mat1[0]) == 0:
        return None  # Handle empty matrices if necessary

    n = len(mat1[0])

    # Dimensions of B: n_b x p
    n_b = len(mat2)
    p = len(mat2[0]) if n_b > 0 else 0

    # 1. Conformability Check: Inner dimensions must match (n == n_b)
    if n != n_b:
        return None

    # The result matrix C will have shape m x p
    result = []

    # 2. Outer Loop (i): Iterate over rows of A (m rows)
    for i in range(m):
        new_row = []

        # 3. Middle Loop (j): Iterate over columns of B (p columns)
        for j in range(p):
            # 4. Inner Loop (k): Calculate the dot product for C[i][j]
            dot_product = 0
            for k in range(n):  # n is the shared dimension
                # C[i][j] += A[i][k] * B[k][j]
                dot_product += mat1[i][k] * mat2[k][j]

            # Add the completed dot product value to the new row
            new_row.append(dot_product)

        # Add the completed row to the result matrix
        result.append(new_row)

    return result
