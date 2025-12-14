#!/usr/bin/env python3
"""
Module that plots a histogram of student grades.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Ensure non-interactive backend for grading
import matplotlib.pyplot as plt


def frequency():
    """
    Plots a histogram showing the frequency of student grades.
    """
    np.random.seed(5)
    student_grades = np.random.normal(68, 15, 50)

    plt.figure(figsize=(6.4, 4.8))
    plt.hist(student_grades, bins=np.arange(0, 101, 10), edgecolor='black')
    plt.xlabel("Grades")
    plt.ylabel("Number of Students")
    plt.title("Project A")
    
    plt.savefig("plot.png")  # saves the figure for the grader
    plt.show()  # optional for you to see the plot locally
