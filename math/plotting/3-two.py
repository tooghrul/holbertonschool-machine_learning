#!/usr/bin/env python3
"""
Task 3: Two Plots
"""
import numpy as np
import matplotlib.pyplot as plt


def two():
    """
    Plots two exponential decay lines (C-14 and Ra-226).
    """
    x = np.arange(0, 21000, 1000)
    r = np.log(0.5)
    t1 = 5730
    t2 = 1600
    y1 = np.exp((r / t1) * x)
    y2 = np.exp((r / t2) * x)
    plt.figure(figsize=(6.4, 4.8))

    # Plot y1 (C-14) as dashed red line
    plt.plot(x, y1, 'r--', label='C-14')
    
    # Plot y2 (Ra-226) as solid green line
    plt.plot(x, y2, 'g-', label='Ra-226')
    
    # Set labels, title, and limits
    plt.xlabel('Time (years)')
    plt.ylabel('Fraction Remaining')
    plt.title('Exponential Decay of Radioactive Elements')
    plt.xlim(0, 20000)
    plt.ylim(0, 1)
    
    # Add legend to upper right corner
    plt.legend(loc='upper right')
    plt.show()
