#!/usr/bin/env python3
"""
Module: 9-sum_total.py
This module is a part of 'Calculus' project.
"""

def summation_i_squared(n):
    """Computes the sum of squares from 1 to n recursively."""
    # Base Case: if n is 0, the sum is 0
    if n <= 0:
        return 0
    
    # Recursive Step
    return n**2 + summation_i_squared(n - 1)
