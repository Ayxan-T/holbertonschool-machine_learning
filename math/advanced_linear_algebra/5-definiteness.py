#!/usr/bin/env python3
"""
Docstring for '5-definiteness' module
"""

import numpy as np


def definiteness(matrix):
    """
    Docstring for 'definiteness' function
    """
    if not isinstance(matrix, np.ndarray):
        raise TypeError("matrix must be a numpy.ndarray")

    shape = matrix.shape
    try:
        is_square = shape[0] == shape[1]
    except IndexError:
        return None

    if not is_square:
        return None

    dets = []
    for i in range(shape[0]):
        dets.append(np.linalg.det(matrix[:i+1, :i+1]))

    # checking positive definiteness
    check_successful = True
    for elm in dets:
        if elm <= 0:
            check_successful = False
            break
    if check_successful:
        return 'Positive definite'

    # checking negative definiteness
    check_successful = True
    for idx, elm in enumerate(dets):
        if (idx % 2 == 0 and elm <= 0) or
        idx % 2 == 1 and elm >= 0:
            check_successful = False
            break
    if check_successful:
        return 'Negative definite'

    # checking indefiniteness
    if dets[-1] != 0:
        return 'Indefinite'

    return None
