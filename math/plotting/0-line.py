#!/usr/bin/env python3
"""
    Module: 0-line
    This module is a part of 'Plotting' project.
"""
import numpy as np
import matplotlib.pyplot as plt


def line():
    """
        plots a cubic line
    """

    y = np.arange(0, 11) ** 3
    plt.figure(figsize=(6.4, 4.8))

    x = np.arange(0, 11)

    plt.plot(x, y)
