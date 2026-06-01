"""
Lightweight EKF Gym environment for fast RL training.
will use numpy here for speed.
"""

import gymnasium as gym


class LightweightEKFEnv(gym.Env):

    def __init__(self, config: Optional[dict] = None):
        super().__init__()
        