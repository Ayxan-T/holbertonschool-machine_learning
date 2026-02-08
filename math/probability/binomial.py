#!/usr/bin/env python3
"""
Module: binomial
"""


class Binomial:
    def __init__(self, data=None, n=1, p=0.5):
        if data is None:
            # Validate provided n and p
            if n <= 0:
                raise ValueError("n must be a positive value")
            if 1 < p or p < 0:
                raise ValueError("p must be greater than 0 and less than 1")
            self.n = int(n)
            self.p = float(p)
        else:
            # Validate data list
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")

            # Calculate Mean and Variance
            mean = sum(data) / len(data)
            variance = sum((x - mean)**2 for x in data) / len(data)

            # Estimate p, then n, then recalculate p
            # p = 1 - (variance / mean)
            estimated_p = 1 - (variance / mean)
            estimated_n = round(mean / estimated_p)
            
            # Recalculate p based on the rounded n
            self.n = int(estimated_n)
            self.p = float(mean / self.n)

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
