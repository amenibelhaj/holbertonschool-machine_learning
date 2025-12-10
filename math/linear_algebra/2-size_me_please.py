#!/usr/bin/env python3
def matrix_shape(matrix):
    shape = []
    current_level = matrix
    while isinstance(current_level, list):
        shape.append(len(current_level))
        if len(current_level) == 0:
            break
        else:
            current_level = current_level[0]
    return shape
