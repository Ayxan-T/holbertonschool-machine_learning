#!/usr/bin/env python3
"""Module: 0-load_env"""

import gymnasium as gym


def load_frozen_lake(desc=None, map_name=None, is_slippery=False):
    """Load a FrozenLake environment.

    Args:
        desc (list[list[str]] | None): Custom map description.
        map_name (str | None): Name of a pre-made map.
        is_slippery (bool): Whether the lake is slippery.

    Returns:
        gymnasium.Env: The FrozenLake environment.
    """
    return gym.make(
        "FrozenLake-v1",
        desc=desc,
        map_name=map_name,
        is_slippery=is_slippery,
    )
