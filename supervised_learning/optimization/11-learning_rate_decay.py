#!/usr/bin/env pyhton3
"""Module: 11-learning_rate_decay"""

import numpy as np


def learning_rate_decay(alpha, decay_rate, global_step, decay_step):
    """Updates the learning rate using inverse time decay in numpy vecorized
    fashion."""
    step = np.asarray(global_step)
    multiplier = np.abs((step + 1) / decay_step)
    return alpha / (1 + decay_rate * multiplier)
