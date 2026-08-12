#!/usr/bin/env python3
"""Module: 2-epsilon_greedy"""

import numpy as np


def epsilon_greedy(Q, state, epsilon):
    """Performs epsilon-greedy to determine next action.
    
    Args:
        Q: (numpy.ndarray) the q-table
        state: current state
        epislon
    
    Returns:
        action_idx: index of next action
    """

    num_actions = Q.shape[1]
    p = np.random.uniform()
    if p > epsilon:
        # find idx of best action col for state
        action_idx = np.argmax(Q[state, :])
    else:
        action_idx = np.random.randint(num_actions)

    return action_idx
