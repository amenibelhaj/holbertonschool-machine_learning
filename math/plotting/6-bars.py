#!/usr/bin/env python3
"""
Module that plots a stacked bar chart of fruits per person.
"""

import numpy as np
import matplotlib.pyplot as plt


def bars():
    """
    Plots a stacked bar graph for fruit quantities of 3 people.
    """
    np.random.seed(5)
    fruit = np.random.randint(0, 20, (4, 3))  # 4 fruits, 3 people

    plt.figure(figsize=(6.4, 4.8))

    # Names of people
    people = ['Farrah', 'Fred', 'Felicia']

    # Colors for each fruit
    colors = ['red', 'yellow', '#ff8000', '#ffe5b4']

    # Labels for legend
    fruit_labels = ['Apples', 'Bananas', 'Oranges', 'Peaches']

    # Stack bars
    bottom = np.zeros(3)  # start at 0 for stacking
    for i in range(fruit.shape[0]):
        plt.bar(people, fruit[i], bottom=bottom, color=colors[i],
                width=0.5, label=fruit_labels[i])
        bottom += fruit[i]  # update bottom for next layer

    # Labels, title, and Y-axis settings
    plt.ylabel("Quantity of Fruit")
    plt.ylim(0, 80)
    plt.yticks(np.arange(0, 81, 10))
    plt.title("Number of Fruit per Person")
    plt.legend()

    # Optional: plt.show() if testing locally
    # plt.show()
