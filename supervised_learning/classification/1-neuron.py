#!/usr/bin/env python3
"""
Module: 0-neuron
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
        @property
        def W(self):
            return self.__W

        self.__b = 0
        @property
        def b(self):
            return self.__b

        self.__A = 0
        @property
        def A(self):
            return self.__A
