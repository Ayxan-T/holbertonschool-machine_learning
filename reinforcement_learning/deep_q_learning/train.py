import numpy as np
import cv2
import gymnasium as gym

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Flatten, Convolution2D, Permute
from tensorflow.keras.optimizers import Adam

from rl.agents.dqn import DQNAgent
from rl.policy import EpsGreedyQPolicy
from rl.memory import SequentialMemory

# ==============================================================================
# 1. ENVIRONMENT WRAPPERS
# Gymnasium environments return 5 values from step() and 2 from reset(), but 
# keras-rl2 expects the older OpenAI Gym interface (4 values and 1 value).
# Atari also requires preprocessing (downsampling to 84x84 and grayscaling).
# ==============================================================================

class AtariKerasRLWrapper(gym.Wrapper):
    """
    Custom wrapper to adapt Gymnasium's Atari environment to keras-rl2:
    - Preprocesses raw 210x160 RGB frames to 84x84 Grayscale.
    - Truncates Gymnasium's 5-tuple step() output to keras-rl2's expected 4-tuple.
    - Truncates Gymnasium's 2-tuple reset() output to keras-rl2's expected 1-tuple.
    """
    def __init__(self, env):
        super().__init__(env)
        # Define the preprocessed observation space (84x84 single-channel uint8 image)
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(84, 84), dtype=np.uint8
        )

    def _preprocess_frame(self, frame):
        # Convert RGB image to Grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        # Resize to 84x84 pixels standard DeepMind Atari size
        resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
        return resized

    def reset(self, **kwargs):
        # Gymnasium reset() returns (obs, info). keras-rl2 only expects obs.
        obs, _ = self.env.reset(**kwargs)
        return self._preprocess_frame(obs)

    def step(self, action):
        # Gymnasium step() returns (obs, reward, terminated, truncated, info).
        # keras-rl2 expects (obs, reward, done, info).
        obs, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        return self._preprocess_frame(obs), reward, done, info


def make_breakout_env():
    # Load Atari Breakout using gymnasium. render_mode is None during training for speed.
    env = gym.make("ALE/Breakout-v5", render_mode=None)
    # Wrap the environment for compatibility and preprocessing
    env = AtariKerasRLWrapper(env)
    return env


# ==============================================================================
# 2. MAIN TRAINING LOGIC
# ==============================================================================
if __name__ == "__main__":
    env = make_breakout_env()
    nb_actions = env.action_space.n

    # Define frame stack window length (feed 4 consecutive frames so DQN sees movement/velocity)
    WINDOW_LENGTH = 4
    input_shape = (WINDOW_LENGTH, 84, 84)

    # ==========================================================================
    # 3. DEEP Q-NETWORK (DQN) ARCHITECTURE
    # Standard Nature DeepMind CNN Architecture for Atari games.
    # ==========================================================================
    model = Sequential()
    
    # keras-rl2 stacks frames across channels, permuting ensures image dimensions are correct
    model.add(Permute((2, 3, 1), input_shape=input_shape))
    
    # Convolutional layers to process spatial visual information
    model.add(Convolution2D(32, (8, 8), strides=(4, 4), activation='relu'))
    model.add(Convolution2D(64, (4, 4), strides=(2, 2), activation='relu'))
    model.add(Convolution2D(64, (3, 3), strides=(1, 1), activation='relu'))
    
    # Flatten features into fully connected layers to compute Q-values for each action
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dense(nb_actions, activation='linear'))

    # ==========================================================================
    # 4. REPLAY MEMORY & EXPLORATION POLICY
    # ==========================================================================
    # Replay memory stores past experiences (states, actions, rewards) to train offline
    memory = SequentialMemory(limit=100000, window_length=WINDOW_LENGTH)

    # Epsilon-Greedy policy: agent explores randomly with high probability early on
    # and gradually transitions to choosing optimal Q-values as training progresses.
    policy = EpsGreedyQPolicy(eps=0.1)

    # ==========================================================================
    # 5. INITIALIZE & TRAIN THE DQN AGENT
    # ==========================================================================
    dqn = DQNAgent(
        model=model,
        nb_actions=nb_actions,
        policy=policy,
        memory=memory,
        nb_steps_warmup=50000,    # Random action steps before training starts to fill memory
        gamma=0.99,               # Discount factor for future rewards
        target_model_update=10000, # Steps interval to update target Q-network weights
        train_interval=4          # Perform gradient step every 4 frames
    )

    # Compile the agent with Adam optimizer and Huber Loss metric
    dqn.compile(Adam(learning_rate=0.00025), metrics=['mae'])

    # Start training loop (e.g. 1 million total environment steps)
    dqn.fit(env, nb_steps=1000000, visualize=False, verbose=2)

    # Save trained policy weights to h5 file
    dqn.save_weights('policy.h5', overwrite=True)
    print("Training finished! Model weights saved to policy.h5.")