#!/usr/bin/env python3
"""
Task 2: Change Scale
"""
import numpy as np
import matplotlib.pyplot as plt


def change_scale():
    """
    Plots exponential decay of C-14 with a logarithmic y-axis.
    """
    x = np.arange(0, 28651, 5730)
    r = np.log(0.5)
    t = 5730
    y = np.exp((r / t) * x)
    plt.figure(figsize=(6.4, 4.8))

    # Plot data
    plt.plot(x, y)
    
    # Set y-axis to logarithmic scale
    plt.yscale('log')
    
    # Set labels, title, and x-axis limit
    plt.xlabel('Time (years)')
    plt.ylabel('Fraction Remaining')
    plt.title('Exponential Decay of C-14')
    plt.xlim(0, 28650)
    plt.show()
