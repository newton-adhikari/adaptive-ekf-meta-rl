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
        noise_profile: Optional[SensorNoiseProfile] = None,
        spread_factor: float = 10.0,
        state_dim: int = 6,
        meas_dim: int = 2,
    ):
        self.profile = noise_profile or SensorNoiseProfile()
        self.spread_factor = spread_factor
        self.state_dim = state_dim
        self.meas_dim = meas_dim

        # Derive nominal Q/R from sensor profile
        self.Q_nominal = self._derive_process_noise()
        self.R_nominal = self._derive_measurement_noise()

    def _derive_process_noise(self) -> np.ndarray:
        """Derive process noise covariance from sensor profile.

        State: [px, py, theta, vx, vy, omega]
        """
        p = self.profile
        return np.array([
            p.odom_linear_noise ** 2,    # px
            p.odom_linear_noise ** 2,    # py
            p.imu_arw ** 2,              # theta
            p.imu_accel_noise ** 2,      # vx
            p.imu_accel_noise ** 2,      # vy
            p.imu_bias_instability ** 2, # omega
        ])

    def _derive_measurement_noise(self) -> np.ndarray:
        """Derive measurement noise from sensor profile.

        Measurement: [px_meas, py_meas] from LiDAR scan matching.
        """
        p = self.profile
        range_noise = p.lidar_range_noise_a  # at nominal range
        return np.array([range_noise ** 2, range_noise ** 2])

    def create_sampler(self) -> TaskSampler:
        """Create a TaskSampler with calibrated ranges."""
        sf = self.spread_factor
        config = {
            "state_dim": self.state_dim,
            "meas_dim": self.meas_dim,
            "q_range": (
                float(np.min(self.Q_nominal) / sf),
                float(np.max(self.Q_nominal) * sf),
            ),
            "r_range": (
                float(np.min(self.R_nominal) / sf),
                float(np.max(self.R_nominal) * sf),
            ),
        }
        return TaskSampler(config)

    def sample_train_tasks(
        self, n: int = 200, seed: int = 42
    ) -> list[TaskConfig]:
        """Sample training task set."""
        rng = np.random.default_rng(seed)
        sampler = self.create_sampler()
        return sampler.sample_batch(n, rng)

    def sample_test_tasks(
        self, n: int = 50, seed: int = 123
    ) -> list[TaskConfig]:
        """Sample held-out test task set (different seed)."""
        rng = np.random.default_rng(seed)
        sampler = self.create_sampler()
        return sampler.sample_batch(n, rng)
