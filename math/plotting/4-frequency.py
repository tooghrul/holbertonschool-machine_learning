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

    # Plot histogram with bins every 10 units and black outline
    plt.hist(student_grades, bins=range(0, 101, 10), edgecolor='black')
    
    # Set labels and title
    plt.xlabel('Grades')
    plt.ylabel('Number of Students')
    plt.title('Project A')
    plt.xlim(0, 100)
    plt.show()
