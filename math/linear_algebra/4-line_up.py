#!/usr/bin/env python3
"""
4-line_up
Module containing the function add_arrays(arr1, arr2)
that adds two lists (arrays) element-wise.
"""


def add_arrays(arr1, arr2):
    """
    Adds two arrays element-wise.

    Args:
        arr1 (list): The first list of ints/floats.
        arr2 (list): The second list of ints/floats.

    Returns:
        list: A new list containing the element-wise sums of arr1 and arr2,
              or None if the arrays are not the same shape.
    """
    # Step 1: Check for conformability (same shape/length)
    if len(arr1) != len(arr2):
        return None

    # Step 2: Perform element-wise addition using list comprehension
    # We iterate from index 0 up to the length of the lists.
    new_array = [arr1[i] + arr2[i] for i in range(len(arr1))]

    return new_array
