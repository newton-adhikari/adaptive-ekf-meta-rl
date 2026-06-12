"""
Sensor characterization: Allan variance analysis, noise model fitting.

This will be used to build calibrated task distributions from real sensor data.
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass
class AllanVarianceResult:
    """Results from Allan variance analysis."""
    taus: np.ndarray           # averaging times
    adev: np.ndarray           # Allan deviation values
    arw: float                 # angle random walk (slope = -0.5)
    bias_instability: float    # bias instability (minimum of adev)
    rrw: float                 # rate random walk (slope = +0.5)

