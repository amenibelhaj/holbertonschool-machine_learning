#!/usr/bin/env python3
"""
6-howdy_partner
Module containing the function cat_arrays(arr1, arr2)
that concatenates two lists (arrays).
"""


def cat_arrays(arr1, arr2):
    """
    Concatenates two arrays (lists) into a new list.

    Args:
        arr1 (list): The first list of ints/floats.
        arr2 (list): The second list of ints/floats.

    Returns:
        list: A new list containing all elements of arr1 followed by
              all elements of arr2.
    """
    # The '+' operator for lists creates a new list by joining the elements.
    return arr1 + arr2
