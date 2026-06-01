"""
This is for task distribution definitions for meta-RL training.
Defines how training tasks are sampled, including calibrated distributions
anchored to real sensor noise models.
"""


import numpy as np
from dataclasses import dataclass
from typing import Optional

from meta_rl.envs.task_sampler import TaskSampler, TaskConfig


@dataclass
class SensorNoiseProfile:
    """Real sensor noise parameters from characterization."""

    # IMU noise (from Allan variance)
    imu_arw: float = 0.005          # angle random walk (rad/s/√Hz)
    imu_bias_instability: float = 0.001  # bias instability (rad/s)
    imu_accel_noise: float = 0.02   # accelerometer noise (m/s²/√Hz)

    # LiDAR noise (range-dependent)
    lidar_range_noise_a: float = 0.01   # constant term (m)
    lidar_range_noise_b: float = 0.001  # linear term (m/m)
    lidar_range_noise_c: float = 0.0    # quadratic term (m/m²)

    # Odometry noise
    odom_linear_noise: float = 0.05     # m/s
    odom_angular_noise: float = 0.02    # rad/s



class CalibratedTaskDistribution:
    """Task distribution centered on real sensor noise with controlled spread.
    """

    def __init__(
        self,
        
    ):
        pass