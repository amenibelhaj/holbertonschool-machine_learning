#!/usr/bin/env python3
"""
Module that plots the exponential decay of Carbon-14
using a logarithmic y-axis scale.
"""

import numpy as np
import matplotlib.pyplot as plt


def change_scale():
    """
    Plots the exponential decay of C-14 over time.
    """
    # Time values (years)
    x = np.arange(0, 28651, 5730)

    # Decay constant
    r = np.log(0.5)
    t = 5730

    # Fraction of C-14 remaining
    y = np.exp((r / t) * x)

    # Create the figure
    plt.figure(figsize=(6.4, 4.8))

    # Plot x versus y as a line graph
    plt.plot(x, y)

    # Label the axes
    plt.xlabel("Time (years)")
    plt.ylabel("Fraction Remaining")

    # Set the title
    plt.title("Exponential Decay of C-14")

    # Set y-axis to logarithmic scale
    plt.yscale("log")

    # Set x-axis limits
    plt.xlim(0, 28650)

    # Display the plot
    plt.show()
