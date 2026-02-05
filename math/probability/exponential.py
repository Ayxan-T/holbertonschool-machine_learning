#!/usr/bin/env python3
"""
Module: exponential
"""


class Exponential:
    def __init__(self, data=None, lambtha=1.):
        self.data = data
        if lambtha <= 0:
            raise ValueError("lambtha must be a positive value")

        if data is None:
            self.lambtha = lambtha
        else:
            if type(data) is not list:
                raise TypeError("data must be a list")

            if len(data) < 2:
                raise ValueError("data must contain multiple values")

            # calculating lambtha (average)
            self.lambtha = 1 / (sum(data) / len(data))

    def fact(n):
        """
        helper
        """
        if n == 0:
            return 1
        res = 1
        for i in range(1, n + 1):
            res *= i

        return res
