#!/usr/bin/env python3
"""Module: 3-q_learning"""

epsilon_greedy = __import__('2-epsilon_greedy').epsilon_greedy
import numpy as np


def train(
        env, Q, episodes=5000, max_steps=100, alpha=0.1, gamma=0.99,
        epsilon=1, min_epsilon=0.1, epsilon_decay=0.05
):

    initial_epsilon = epsilon

    # Need to play 'episodes' times
    for episode in range(episodes):
        state, info = env.reset()
        steps = 0
        # while not terminal state (lake or goal), and not max_steps, play
        while steps < max_steps: 
            # determine the next action and make it
            action = epsilon_greedy(Q, state, epsilon)

            # step in env
            next_state, reward, terminated, truncated, info = env.step(action)

            # update reward if agent falls into hole
            if terminated and reward == 0:
                reward = -1

            # Q-learning update
            best_next = np.max(Q[next_state, :])
            Q[state, action] = Q[state, action] + alpha * (
                reward + gamma * best_next - Q[state, action]
            )

            state = next_state
            steps += 1

            if terminated or truncated:
                break

        # update 'epsilon' by 'epsilon_decay' and continue
        epsilon = min_epsilon + (initial_epsilon - min_epsilon) * \
            np.exp(-epsilon_decay * episode)

    return Q
