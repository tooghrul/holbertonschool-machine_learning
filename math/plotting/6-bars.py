#!/usr/bin/env python3
"""
Task 6: Stacked Bar Graph
"""
import numpy as np
import matplotlib.pyplot as plt


def bars():
    """
    Plots a stacked bar graph of fruit per person.
    """
    np.random.seed(5)
    fruit = np.random.randint(0, 20, (4, 3))
    plt.figure(figsize=(6.4, 4.8))
    people = ['Farrah', 'Fred', 'Felicia']
    colors = ['red', 'yellow', '#ff8000', '#ffe5b4']
    labels = ['apples', 'bananas', 'oranges', 'peaches']
    width = 0.5
    # Initialize bottom tracker for stacking
    bottom = np.zeros(3)
    for i in range(len(fruit)):
        plt.bar(
            people,
            fruit[i],
            width,
            bottom=bottom,
            color=colors[i],
            label=labels[i]
        )
        # Update bottom for the next fruit layer
        bottom += fruit[i]

    plt.ylabel('Quantity of Fruit')
    plt.title('Number of Fruit per Person')
    plt.yticks(np.arange(0, 81, 10))
    plt.legend()
    plt.show()
