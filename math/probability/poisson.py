#!/usr/bin/env python3
"""
Module: poisson
"""


class Poisson:
    """
    Class: Poisson
    """
    def __init__(self, data=None, lambtha=1.):
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
            Sum = 0
            for elm in data:
                Sum += elm
            self.lambtha = Sum / len(data)

    def pmf(self, k):
        count = 0
        for elm in data:
            if elm == k:
                count += 1
        
        return count / Sum
