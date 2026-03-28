#!/usr/bin/env python3
"""
Module to determine if gradient descent should stop early
"""


def early_stopping(cost, opt_cost, threshold, patience, count):
    """
    Determines if you should stop gradient descent early

    Args:
        cost: current validation cost
        opt_cost: lowest recorded validation cost
        threshold: threshold used for early stopping
        patience: patience count used for early stopping
        count: count of how long the threshold has not been met

    Returns:
        A boolean (stop early or not) and the updated count
    """
    # Check if the current cost is better than the optimal cost
    # by more than the threshold
    if opt_cost - cost > threshold:
        count = 0
    else:
        count += 1

    # If count reaches patience, we stop (True)
    if count >= patience:
        return True, count

    return False, count
