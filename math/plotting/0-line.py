#!/usr/bin/env python3
"""
Module that plots a line graph.
"""

import numpy as np
import matplotlib.pyplot as plt


def line():
    """
    Plots y = x^3 as a solid red line graph.

    The x-axis ranges from 0 to 10.
    """
    # Create y values: y = x^3 for x from 0 to 10
    y = np.arange(0, 11) ** 3

    # Create x values from 0 to 10
    x = np.arange(0, 11)

    # Create a figure with the default size
    plt.figure(figsize=(6.4, 4.8))

    # Plot y against x as a solid red line
    plt.plot(x, y, 'r-')

    # Set the limits of the x-axis
    plt.xlim(0, 10)

    # Display the plot
    plt.show()
