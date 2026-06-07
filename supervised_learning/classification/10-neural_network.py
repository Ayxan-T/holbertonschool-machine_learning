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

        # Initialize private parameters in hidden layer
        self.__W1 = np.random.normal(size=(nodes, nx))
        self.__b1 = np.zeros(shape=(nodes, 1))
        self.__A1 = 0     # activations

        # Initialize private parameters in output neuron
        self.__W2 = np.random.normal(size=(1, nodes))
        self.__b2 = 0
        self.__A2 = 0

    @property
    def W1(self):
        return self.__W1

    @property
    def b1(self):
        return self.__b1

    @property
    def A1(self):
        return self.__A1

    @property
    def W2(self):
        return self.__W2

    @property
    def b2(self):
        return self.__b2

    @property
    def A2(self):
        return self.__A2
    
    def forward_prop(self, X):
        """ forward_prop(X)
        
        X.shape: (nx, m)
        """

        Z1 = np.matmul(self.__W1, X) + self.__b1  # (nodes, m)
        self.__A1 = 1 / (1 + np.exp(-Z1))

        Z2 = np.matmul(self.__W2, self.__A1) + self.__b2  # (1, m)
        self.__A2 = 1 / (1 + np.exp(-Z2))

        return self.__A1, self.__A2
