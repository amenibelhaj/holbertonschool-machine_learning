#!/usr/bin/env python3
"""
Module that plots all previous plots in one figure.
"""

import numpy as np
import matplotlib.pyplot as plt


def all_in_one():
    """
    Plots all 5 previous graphs in a 3x2 grid, with the last plot spanning two columns.
    All axis labels and titles are x-small.
    """
    # -----------------------------
    # Data preparation
    # -----------------------------
    # Task 0: Line
    x0 = np.arange(0, 11)
    y0 = x0 ** 3

    # Task 1: Scatter
    mean = [69, 0]
    cov = [[15, 8], [8, 15]]
    np.random.seed(5)
    x1, y1 = np.random.multivariate_normal(mean, cov, 2000).T
    y1 += 180

    # Task 2: C-14 decay
    x2 = np.arange(0, 28651, 5730)
    r2 = np.log(0.5)
    t2 = 5730
    y2 = np.exp((r2 / t2) * x2)

    # Task 3: Two decays
    x3 = np.arange(0, 21000, 1000)
    r3 = np.log(0.5)
    t31 = 5730
    t32 = 1600
    y31 = np.exp((r3 / t31) * x3)
    y32 = np.exp((r3 / t32) * x3)

    # Task 4: Histogram
    np.random.seed(5)
    student_grades = np.random.normal(68, 15, 50)

    # -----------------------------
    # Create figure and subplots
    # -----------------------------
    fig = plt.figure(figsize=(12, 12))
    fig.suptitle("All in One")

    # Task 0: Line
    ax0 = fig.add_subplot(3, 2, 1)
    ax0.plot(x0, y0, 'r-')
    ax0.set_xlabel("x", fontsize='x-small')
    ax0.set_ylabel("y", fontsize='x-small')
    ax0.set_title("Line", fontsize='x-small')

    # Task 1: Scatter
    ax1 = fig.add_subplot(3, 2, 2)
    ax1.scatter(x1, y1, c='m')
    ax1.set_xlabel("Height (in)", fontsize='x-small')
    ax1.set_ylabel("Weight (lbs)", fontsize='x-small')
    ax1.set_title("Scatter", fontsize='x-small')

    # Task 2: C-14 decay
    ax2 = fig.add_subplot(3, 2, 3)
    ax2.plot(x2, y2)
    ax2.set_xlabel("Time (years)", fontsize='x-small')
    ax2.set_ylabel("Fraction Remaining", fontsize='x-small')
    ax2.set_title("C-14 Decay", fontsize='x-small')
    ax2.set_yscale('log')

    # Task 3: Two decays
    ax3 = fig.add_subplot(3, 2, 4)
    ax3.plot(x3, y31, 'r--', label='C-14')
    ax3.plot(x3, y32, 'g-', label='Ra-226')
    ax3.set_xlabel("Time (years)", fontsize='x-small')
    ax3.set_ylabel("Fraction Remaining", fontsize='x-small')
    ax3.set_title("Two Decays", fontsize='x-small')
    ax3.set_xlim(0, 20000)
    ax3.set_ylim(0, 1)
    ax3.legend(fontsize='x-small', loc='upper right')

    # Task 4: Histogram (spanning two columns)
    ax4 = fig.add_subplot(3, 1, 3)
    ax4.hist(student_grades, bins=np.arange(0, 101, 10), edgecolor='black')
    ax4.set_xlabel("Grades", fontsize='x-small')
    ax4.set_ylabel("Number of Students", fontsize='x-small')
    ax4.set_title("Project A", fontsize='x-small')

    # Adjust layout
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    # Optional: plt.show() if you want to see it locally
    # plt.show()
