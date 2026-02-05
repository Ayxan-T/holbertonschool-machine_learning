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

    def fact(n):
        if n == 0:
            return 1
        res = 1
        for i in range(1, n + 1):
            res *= i

        return res

    def pmf(self, k):
        """
        Method: pmf
        """
        if k < 0:
            return 0
        k = int(k)
        e = 2.7182818285
        fac = Poisson.fact(k)
        P = (self.lambtha**k) * e**(-self.lambtha) / fac

        return P

    def cdf(self, k):
        """
        Method: cdf
        """
        if k < 0:
            return 0
        k = int(k)
        cdf = 0
        for i in range(k + 1):
            cdf += self.pmf(i)

        return cdf
        
