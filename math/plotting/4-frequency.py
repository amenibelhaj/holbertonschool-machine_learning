#!/usr/bin/env python3
"""
Module that plots a histogram of student grades.
"""

import numpy as np
import matplotlib.pyplot as plt


def frequency():
    """
    Plots a histogram showing the frequency of student grades.
    """
    # Generate random student grades
    np.random.seed(5)
    student_grades = np.random.normal(68, 15, 50)

    # Create the figure
    plt.figure(figsize=(6.4, 4.8))

    # Plot histogram with bins every 10 units and black edges
    plt.hist(student_grades, bins=range(0, 101, 10), edgecolor='black')

    # Label the axes
    plt.xlabel("Grades")
    plt.ylabel("Number of Students")

    # Set the title
    plt.title("Project A")

    # Display the plot
    plt.show()
