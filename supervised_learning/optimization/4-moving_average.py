#!/usr/bin/env python3
"""Calculates the weighted moving average with bias correction"""


def moving_average(data, beta):
    """
    Calculates the weighted moving average of a data set

    Args:
        data: list of data to calculate the moving average of
        beta: weight used for the moving average (smoothing factor)

    Returns:
        a list containing the moving averages of data
    """
    v = 0
    moving_averages = []

    for i, theta in enumerate(data):
        # Step 1: Calculate the weighted average
        # The index i starts at 0, so time t = i + 1
        t = i + 1
        v = (beta * v) + ((1 - beta) * theta)

        # Step 2: Apply bias correction
        v_corrected = v / (1 - (beta ** t))
        
        moving_averages.append(v_corrected)

    return moving_averages
