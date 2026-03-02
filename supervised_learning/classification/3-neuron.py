#!/usr/bin/env python3
"""
Module: 2-neuron
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
        return (1/m) * np.sum(Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A))
