#!/usr/bin/env python3
""" This module provides a class  that defines a neural network with one hidden
layer performing binary classification.

Class: NeuralNetwork
"""

import numpy as np


class NeuralNetwork:
    """ Class: NeuralNetwork """
    def __init__(self, nx, nodes):

        # Validate nx (number of inputs)
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")

        # Validate nodes (number of nodes in layer)
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
        returns:
            A1 (nodes, m)
            A2 (1, m)
        """

        Z1 = np.matmul(self.__W1, X) + self.__b1
        self.__A1 = 1 / (1 + np.exp(-Z1))  # (nodes, m)

        Z2 = np.matmul(self.__W2, self.__A1) + self.__b2
        self.__A2 = 1 / (1 + np.exp(-Z2))  # (1, m)

        return self.__A1, self.__A2

    def cost(self, Y, A):
        """ cost(Y, A)

        Y.shape: (1, m)
        A.shape: (1, m)
        """
        m = Y.shape[1]

        # IN CASE OF CHECKER ERROR: TRY ALSO MAKING FIRST TERM ERROR-PROOF
        # ( np.log(0.0000001 + A) )
        return (-1/m) * np.sum(Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A))

    def evaluate(self, X, Y):
        """ evaluate(X, Y)

        X.shape: (nx, m)
        Y.shape: (1, m)
        """

        A = self.forward_prop(X)[1]
        preds = np.where(0.5 <= A, 1, 0)
        cost = self.cost(Y, A)

        return preds, cost

    def gradient_descent(self, X, Y, A1, A2, alpha=0.05):
        """
        W2: (1, nodes)
        A1: (nodes, m)
        A2: (1, m)
        Y: (1, m)
        """
        m = Y.shape[1]
    
        dA2 = np.where(Y == 1, -1 / A2, 1 / (1 - A2))  # (1, m)
        dZ2 = dA2 * A2 * (1 - A2)  # unreduced (1, m)
        db2 = (1 / m) * np.sum(dZ2, axis=1, keepdims=True)  # (1, 1)
        dW2 = (1 / m) * np.matmul(dZ2, A1.T)  # (1, nodes)

        self.__b2 -= alpha * db2
        self.__W2 -= alpha * dW2
