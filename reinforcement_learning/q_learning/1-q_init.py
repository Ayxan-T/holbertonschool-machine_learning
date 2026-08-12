#!/usr/bin/env python3
"""Module: 1-q_init"""

import numpy as np


def q_init(env):
    """Initialize the Q-table for a FrozenLake environment.

    Args:
        env: The FrozenLakeEnv instance.

    Returns:
        numpy.ndarray (n_observations, n_actions): Q-table initialized to zeros
    """
    return np.zeros((env.observation_space.n, env.action_space.n))
