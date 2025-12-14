#!/usr/bin/env python3
"""
Module that plots two exponential decay curves
for different radioactive elements.
"""

import numpy as np
import matplotlib.pyplot as plt


def two():
    """
    Plots the exponential decay of C-14 and Ra-226
    on the same graph for comparison.
    """
    # Time values (years)
    x = np.arange(0, 21000, 1000)

    # Decay constant
    r = np.log(0.5)

    # Half-lives
    t1 = 5730   # Carbon-14
    t2 = 1600   # Radium-226

    # Fraction remaining for each element
    y1 = np.exp((r / t1) * x)
    y2 = np.exp((r / t2) * x)

    # Create the figure
    plt.figure(figsize=(6.4, 4.8))

    # Plot C-14 as a dashed red line
    plt.plot(x, y1, 'r--', label='C-14')

    # Plot Ra-226 as a solid green line
    plt.plot(x, y2, 'g-', label='Ra-226')

    # Label the axes
    plt.xlabel("Time (years)")
    plt.ylabel("Fraction Remaining")

    # Set the title
    plt.title("Exponential Decay of Radioactive Elements")

    # Set axis limits
    plt.xlim(0, 20000)
    plt.ylim(0, 1)

    # Add legend in the upper right corner
    plt.legend(loc='upper right')

    # Display the plot
    plt.show()
