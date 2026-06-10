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

