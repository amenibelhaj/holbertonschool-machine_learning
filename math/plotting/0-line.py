#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

def line():
    # y is calculated as [0^3, 1^3, ..., 10^3]
    y = np.arange(0, 11) ** 3
    plt.figure(figsize=(6.4, 4.8))

    # your code here
    # 1. Use plt.plot()
    # 2. Pass 'y' as the data. (When the x-data is omitted, it defaults to the indices: 0, 1, ..., 10)
    # 3. Use the format string 'r-' for a solid red line.
    plt.plot(y, 'r-')
    
    # Optional: Display the plot. In a script environment, this would show the window.
    # If the environment saves the image automatically, this line may not be needed.
    plt.show()
