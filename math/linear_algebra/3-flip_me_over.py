#!/usr/bin/env python3
def matrix_transpose(matrix):
    """
    Returns the transpose of a 2D matrix.

    Args:
        matrix (list of lists): The 2D matrix to be transposed.
                                Assumed to be non-empty and uniform.

    Returns:
        list of lists: A new matrix representing the transpose of the input matrix.
    """
    # 1. Determine the dimensions of the original matrix
    # The number of rows (m) is the length of the matrix.
    num_rows = len(matrix)
    
    # The number of columns (n) is the length of the first row (since it's uniform).
    # Since the matrix is guaranteed to be non-empty, matrix[0] is safe.
    num_cols = len(matrix[0])
    
    # The resulting transpose matrix will have shape num_cols x num_rows.
    
    # 2. Use nested list comprehensions to build the new matrix
    # The outer loop iterates over the new rows (which were the old columns)
    # The inner loop iterates over the new columns (which were the old rows)
    transposed_matrix = [
        # Iterate over the old rows (i) to form the new columns
        [matrix[i][j] for i in range(num_rows)] 
        
        # Iterate over the old columns (j) to form the new rows
        for j in range(num_cols)
    ]
    
    return transposed_matrix
