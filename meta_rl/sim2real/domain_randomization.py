"""
Need to Calibrate domain randomization for sim-to-real transfer.

PLAN::::
The training task distribution is taken as real sensor noise models
collected on the target platform, not arbitrary ranges.
"""


import numpy as np
from typing import Optional
from dataclasses import dataclass

from meta_rl.tasks.task_distribution import SensorNoiseProfile


@dataclass
class DomainRandomizationConfig:
    """Configuration for calibrated domain randomization."""

    # Noise scaling ranges (multiplicative, log-uniform)
    q_scale_range: tuple[float, float] = (0.1, 10.0)
    r_scale_range: tuple[float, float] = (0.1, 10.0)

    # Dynamics perturbations
    mass_scale_range: tuple[float, float] = (0.8, 1.2)
    friction_scale_range: tuple[float, float] = (0.5, 2.0)
    wheel_radius_scale_range: tuple[float, float] = (0.95, 1.05)

    # Sensor perturbations
    imu_bias_range: tuple[float, float] = (-0.01, 0.01)  # rad/s
    lidar_offset_range: tuple[float, float] = (-0.02, 0.02)  # m

    # Environment perturbations
    terrain_roughness_range: tuple[float, float] = (0.0, 0.05)  # m



class CalibratedDomainRandomizer:
    """
    Generates randomized simulation parameters according to real data.

    """

    def __init__(
        self,
        noise_profile: Optional[SensorNoiseProfile] = None,
        config: Optional[DomainRandomizationConfig] = None,
    ):
        self.profile = noise_profile or SensorNoiseProfile()
        self.config = config or DomainRandomizationConfig()

    def sample_noise_params(
        self, rng: Optional[np.random.Generator] = None
    ) -> dict:
        """Sample randomized noise parameters centered on real profile.

        """
        rng = rng or np.random.default_rng()
        cfg = self.config
        p = self.profile

        # Process noise: scale each component independently (log-uniform)
        q_scales = np.exp(rng.uniform(
            np.log(cfg.q_scale_range[0]),
            np.log(cfg.q_scale_range[1]),
            size=6,
        ))
        Q_diag = np.array([
            p.odom_linear_noise ** 2,
            p.odom_linear_noise ** 2,
            p.imu_arw ** 2,
            p.imu_accel_noise ** 2,
            p.imu_accel_noise ** 2,
            p.imu_bias_instability ** 2,
        ]) * q_scales

        # Measurement noise
        r_scales = np.exp(rng.uniform(
            np.log(cfg.r_scale_range[0]),
            np.log(cfg.r_scale_range[1]),
            size=2,
        ))
        R_diag = np.array([
            p.lidar_range_noise_a ** 2,
            p.lidar_range_noise_a ** 2,
        ]) * r_scales

        return {"Q_diag": Q_diag, "R_diag": R_diag}

    def sample_dynamics_params(
        self, rng: Optional[np.random.Generator] = None
    ) -> dict:
        """Sample randomized dynamics parameters."""
        rng = rng or np.random.default_rng()
        cfg = self.config

        return {
            "mass_scale": rng.uniform(*cfg.mass_scale_range),
            "friction_scale": rng.uniform(*cfg.friction_scale_range),
            "wheel_radius_scale": rng.uniform(*cfg.wheel_radius_scale_range),
        }

    def sample_sensor_perturbations(
        self, rng: Optional[np.random.Generator] = None
    ) -> dict:
        """Sample sensor bias/offset perturbations."""
        rng = rng or np.random.default_rng()
        cfg = self.config

        return {
            "imu_gyro_bias": rng.uniform(*cfg.imu_bias_range, size=3),
            "imu_accel_bias": rng.uniform(*cfg.imu_bias_range, size=3),
            "lidar_range_offset": rng.uniform(*cfg.lidar_offset_range),
        }

    def sample_full_config(
        self, rng: Optional[np.random.Generator] = None
    ) -> dict:
        """Sample a complete randomized configuration."""
        rng = rng or np.random.default_rng()
        return {
            **self.sample_noise_params(rng),
            **self.sample_dynamics_params(rng),
            **self.sample_sensor_perturbations(rng),
            "terrain_roughness": rng.uniform(
                *self.config.terrain_roughness_range
            ),
        }
