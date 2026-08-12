#!/usr/bin/env python3
"""Module: 4-play"""

import numpy as np


def play(env, Q, max_steps=100):
    """Play one episode with a trained Q-table.

    Args:
        env: FrozenLakeEnv instance.
        Q: numpy.ndarray containing the Q-table.
        max_steps: Maximum number of steps in the episode.

    Returns:
        total_rewards: Total rewards for the episode.
        renders: List of rendered board state strings.
    """
    total_rewards = 0
    renders = []

    state, info = env.reset()
    render = env.render()
    renders.append(render)

    for _ in range(max_steps):

        # action = np.argmax(Q[state])
        best_actions = np.flatnonzero(Q[state] == Q[state].max())
        action = np.random.choice(best_actions)

        state, reward, terminated, truncated, info = env.step(action)

        total_rewards += reward
        render = env.render()
        renders.append(render)

        if terminated or truncated:
            break

    return total_rewards, renders
