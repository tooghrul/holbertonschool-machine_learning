#!/usr/bin/env python3
"""
Task 4: Frequency
"""
import numpy as np
import matplotlib.pyplot as plt


def frequency():
    """
    Plots a histogram of student grades.
    """
    np.random.seed(5)
    student_grades = np.random.normal(68, 15, 50)
    plt.figure(figsize=(6.4, 4.8))

    # Define bins every 10 units from 0 to 100
    bins = list(range(0, 101, 10))
    plt.xlim(0, 100)
    plt.ylim(0, 30)

    # Plot the histogram
    # edgecolor='black' adds the requested outline
    plt.hist(student_grades, bins=bins, edgecolor='black')

    # Add labels and title
    plt.xlabel('Grades')
    plt.ylabel('Number of Students')
    plt.title('Project A')
    plt.xticks(bins)

    # Display the plot
    plt.show()
