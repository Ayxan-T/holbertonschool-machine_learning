#!/usr/bin/env python3
"""
Module: normal
"""


class Normal:
    """
    Class: Normal
    """
    def __init__(self, data=None, mean=0., stddev=1.):
        if stddev <= 0:
            raise ValueError("stddev must be a positive value")

        if data is not None:
            if type(data) is not list:
                raise TypeError("data must be a list")

            if len(data) < 2:
                raise ValueError("data must contain multiple values")

            mean = sum(data) / len(data)

            sqrd_residuals = [(mean - elm)**2 for elm in data]
            stddev = (sum(sqrd_residuals) / len(data))**0.5

        self.mean = mean
        self.stddev = stddev

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
        return (1 + (2 / pi**0.5) * \
            (z - z**3 / 3 + z**5 / 10 -
            z**7 / 42 + z**9 / 216)) / 2
