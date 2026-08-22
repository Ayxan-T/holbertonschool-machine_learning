import numpy as np
import cv2
import gymnasium as gym

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Convolution2D, Permute
from tensorflow.keras.optimizers import Adam

from rl.agents.dqn import DQNAgent
from rl.policy import GreedyQPolicy
from rl.memory import SequentialMemory

# ==============================================================================
# 1. ENVIRONMENT WRAPPERS (Identical Preprocessing Wrapper)
# ==============================================================================

class AtariKerasRLWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(84, 84), dtype=np.uint8
        )

    def _preprocess_frame(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
        return resized

    def reset(self, **kwargs):
        obs, _ = self.env.reset(**kwargs)
        return self._preprocess_frame(obs)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        return self._preprocess_frame(obs), reward, done, info


# ==============================================================================
# 2. MAIN EVALUATION / VISUALIZATION LOGIC
# ==============================================================================
if __name__ == "__main__":
    # Create gymnasium environment with 'human' render mode so a screen renders
    env = gym.make("ALE/Breakout-v5", render_mode="human")
    env = AtariKerasRLWrapper(env)
    
    nb_actions = env.action_space.n
    WINDOW_LENGTH = 4
    input_shape = (WINDOW_LENGTH, 84, 84)

    # Re-build identical Neural Network architecture
    model = Sequential()
    model.add(Permute((2, 3, 1), input_shape=input_shape))
    model.add(Convolution2D(32, (8, 8), strides=(4, 4), activation='relu'))
    model.add(Convolution2D(64, (4, 4), strides=(2, 2), activation='relu'))
    model.add(Convolution2D(64, (3, 3), strides=(1, 1), activation='relu'))
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dense(nb_actions, activation='linear'))

    # Memory is needed for initial setup in keras-rl, even if only testing
    memory = SequentialMemory(limit=10000, window_length=WINDOW_LENGTH)

    # GreedyQPolicy selects strictly the best action without random exploration
    policy = GreedyQPolicy()

    # Instantiate DQNAgent
    dqn = DQNAgent(
        model=model,
        nb_actions=nb_actions,
        policy=policy,
        memory=memory
    )
    dqn.compile(Adam(learning_rate=0.00025), metrics=['mae'])

    # Load saved weights trained via train.py
    dqn.load_weights('policy.h5')
    print("Loaded policy.h5 weights successfully.")

    # Run evaluation episodes and render output to screen
    dqn.test(env, nb_episodes=5, visualize=True)
    env.close()