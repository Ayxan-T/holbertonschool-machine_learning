#!/usr/bin/env python3
"""
Docstring for '100-slice_like_a_ninja' module
"""

def np_slice(matrix, axes={}):
    """
    Docstring for 'np_slice' function
    """

    slicers = []
    max_key = max(axes.keys())
    for key in range(max_key + 1):
        if key in axes.keys():
            slicers.append(slice(*axes[key])) # Corrected: use axes[key] for slice parameters
        else:
            slicers.append(slice(None))

    return matrix[*tuple(slicers)]
