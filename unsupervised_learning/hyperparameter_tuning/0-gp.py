#!/usr/bin/env python3
""" Module: 0-gp.py """

import numpy as np


class GaussianProcess:
    """ A noiseless 1D Gaussian process. """
    def __init__(self, X_init, Y_init, l=1, sigma_f=1):
        