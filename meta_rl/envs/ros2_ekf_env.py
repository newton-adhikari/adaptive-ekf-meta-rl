"""
Created ROS2/Gazebo-based EKF environment for sim-to-real validation.
Wraps a Gazebo simulation with ROS2 topics as a Gym environment.
"""

class ROS2EKFEnv(gym.Env):
    def __init__(self):
        super().__init__()
        