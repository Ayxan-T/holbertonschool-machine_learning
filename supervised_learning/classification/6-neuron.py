#!/usr/bin/env python3
""" This module defines a class which defines a single neuron performing binary
classification

Class: Neuron
"""

import numpy as np


class Neuron:
    """
    Class: Neuron
    """
    def __init__(self, nx):
        if type(nx) is not int:
            raise TypeError("nx must be an integer")

        if nx < 1:
            raise ValueError("nx must be a positive integer")

        self.__W = np.random.normal(size=(1, nx))
        self.__b = 0
        self.__A = 0

    @property
    def W(self):
        return self.__W

    @property
    def b(self):
        return self.__b

    @property
    def A(self):
        return self.__A

    def forward_prop(self, X):
        """
        X.shape = (nx, m)
        W.shape = (1, nx)
        """
        Z = np.matmul(self.__W, X) + self.__b

        # sigmoid
        self.__A = 1 / (1 + np.exp(-Z))

        return self.__A

    def cost(self, Y, A):
        """
        Y.shape = (1, m)
        """
        m = Y.shape[1]
        return -(1/m) * np.sum(Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A))

    def evaluate(self, X, Y):
        """
        function: evaluate
        """
        preds = self.forward_prop(X)

        cost = self.cost(Y, preds)
        return np.where(preds > 0.5, 1, 0), cost

    def gradient_descent(self, X, Y, A, alpha=0.05):
        """
        Function: gradient_descent
        """
        m = X.shape[1]
        dz = A - Y
        dw = (1 / m) * np.matmul(dz, X.T)
        db = (1 / m) * np.sum(dz)

        self.__W -= alpha * dw
        self.__b -= alpha * db

    def train(self, X, Y, iterations=5000, alpha=0.05):
        """
        Function: train
        """
        # X: (nx, m);   Y: (1, m);

        # Validate iterations
        if type(iterations) is not int:
            raise TypeError("iterations must be an integer")
        if iterations < 1:
            raise ValueError("iterations must be a positive integer")

        # Validate alpha
        if type(alpha) is not float:
            raise TypeError("alpha must be a float")
        if alpha < 0:
            raise ValueError("alpha must be positive")

        for i in range(iterations):
            # Compute Activations
            A = self.forward_prop(X)

            # Do gradient decsent update
            self.gradient_descent(X, Y, A, alpha)

        return self.evaluate(X, Y)
