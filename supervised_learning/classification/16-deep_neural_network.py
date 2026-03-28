#!/usr/bin/env python3
""" This module will define a class named DeepNeuralNetwork """
import numpy as np


class DeepNeuralNetwork:
    """ Deep NN """
    def __init__(self, nx, layers):
        """ Initialize class """
        # validate nx
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")

        if nx < 1:
            raise ValueError("nx must be a positive integer")

        # validate layers
        if not isinstance(layers, list) or len(layers) == 0:
            raise TypeError("layers must be a list of positive integers")

        self.L = len(layers)
        self.cache = {}
        self.weights = {}

        prev = nx

        for i, nodes in enumerate(layers):

            # validate layer size
            if not isinstance(nodes, int) or nodes <= 0:
                raise TypeError("layers must be a list of positive integers")

            # He initialization
            self.weights["W{}".format(i + 1)] = (
                np.random.randn(nodes, prev) * np.sqrt(2 / prev)
            )

            # bias initialization
            self.weights["b{}".format(i + 1)] = np.zeros((nodes, 1))

            prev = nodes
