#!/usr/bin/env python3
"""
Module: binomial
"""


class Binomial:
    """
    Class: Binomial
    """
    def __init__(self, data=None, n=1., p=0.5):
        if n <= 0:
            raise ValueError("n must be a positive value")

        if p < 0 or p > 1:
            raise ValueError("p must be greater than 0 and less than 1")

        if data is not None:
            if type(data) is not list:
                raise TypeError("data must be a list")

            if len(data) < 2:
                raise ValueError("data must contain multiple values")

            mean = sum(data) / len(data)

            sqrd_residuals = [(mean - elm)**2 for elm in data]
            var = sum(sqrd_residuals) / len(data)

            p = 1 - mean / var
            n = mean / p
            
            p = mean / n

            temp = dict()
            for elm in data:
                if temp.get(elm, None) is None:
                    temp[elm] = 1
                else:
                    temp[elm] += 1


        self.p = float(p)
        self.n = int(n)

    def z_score(self, x):
        """
        Method: z_score
        """
        return (x - self.mean) / self.stddev

    def x_value(self, z):
        """
        Method: x_value
        """
        return self.mean + z * self.stddev

    def pdf(self, x):
        """
        Method: pdf
        """
        e = 2.7182818285
        pi = 3.1415926536
        return (1 / (self.stddev * (2 * pi)**0.5)) * \
            e**(-0.5 * ((x - self.mean) / self.stddev)**2)

    def cdf(self, x):
        """
        Method: cdf
        """
        pi = 3.1415926536
        z = (x - self.mean) / (self.stddev * 2**0.5)
        return (1 + (2 / pi**0.5) *
                (z - z**3 / 3 + z**5 / 10 -
                 z**7 / 42 + z**9 / 216)) / 2
