#!/usr/bin/env python3
""" This module provides a class  that defines a neural network with one hidden
layer performing binary classification.

Class: NeuralNetwork
"""

import numpy as np


class NeuralNetwork:
    """ Class: NeuralNetwork """
    def __init__(self, nx, nodes):

        # Validate nx
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")

        # Validate nodes
        if type(nodes) is not int:
            raise TypeError("nodes must be an integer")
        if nodes < 1:
            raise ValueError("nodes must be a positive integer")

        # Initialize parameters in hidden layer
        self.W1 = np.random.normal(size=(nodes, nx))
        self.b1 = np.zeros(shape=(nodes, 1))
        self.A1 = 0 # activations

        # Initialize parameters in output neuron
        self.W2 = np.random.normal(size=(1, nodes))
        self.b2 = 0
        self.A2 = 0
