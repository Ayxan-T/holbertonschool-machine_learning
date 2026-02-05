#!/usr/bin/env python3
"""
Module: poisson
"""


class Poisson:
    """
    Class: Poisson
    """
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
            self.lambtha = sum(data) / len(data)

    def pmf(self, k):
        e = 2.7182818285
        fac = k
        for i in range(1, k):
            fac *= i
        P = (self.lambtha**k) * e**(-self.lambtha) / fac

        return P
