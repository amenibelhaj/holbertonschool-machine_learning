#!/usr/bin/env python3
"""
Module that plots a scatter graph of men's height vs weight.
"""

import numpy as np
import matplotlib.pyplot as plt


def scatter():
    """
    Plots a scatter plot of height (inches) versus weight (pounds).
    """
    mean = [69, 0]
    cov = [[15, 8], [8, 15]]

    # Set random seed for reproducibility
    np.random.seed(5)

    # Generate random height (x) and weight (y) data
    x, y = np.random.multivariate_normal(mean, cov, 2000).T
    y += 180

    # Create the figure
    plt.figure(figsize=(6.4, 4.8))

    # Plot the data as magenta points
    plt.scatter(x, y, c='m')

    # Label the axes
    plt.xlabel("Height (in)")
    plt.ylabel("Weight (lbs)")

    # Set the title
    plt.title("Men's Height vs Weight")

    # Display the plot
    plt.show()
