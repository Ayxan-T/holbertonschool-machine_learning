#!/usr/bin/env python3
"""Module: 4-positional_encoding"""

import numpy as np


def positional_encoding(max_seq_len, dm):
    """Calculates positional encoding for a transformer."""
    # Create position indices array of shape (max_seq_len, 1)
    pos = np.arange(max_seq_len)[:, np.newaxis]

    # Create dimension indices array for the denominator (1, dm)
    i = np.arange(dm)[np.newaxis, :]

    # Calculate angle rates using 2 * (i // 2) / dm
    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / dm)

    # Compute angle argument matrix of shape (max_seq_len, dm)
    angle_rads = pos * angle_rates

    # Apply sine to even indices (2i)
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

    # Apply cosine to odd indices (2i + 1)
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

    return angle_rads