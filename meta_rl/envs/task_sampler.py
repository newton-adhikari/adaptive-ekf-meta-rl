"""
Task distribution sampler for meta-RL training.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional

@dataclass
class TaskConfig:
    """ to define noise profile and trajectory."""

    Q_base: np.ndarray          # Base process noise diagonal
    R_base: np.ndarray          # Base measurement noise diagonal
    noise_regime: str           # 'stationary', 'abrupt_change', 'slow_drift', 'periodic'
    trajectory_type: str        # 'circle', 'figure8', 'straight', 'random_walk'
    change_time: Optional[int] = None   # Timestep of abrupt change
    Q_after: Optional[np.ndarray] = None  # Q after change (abrupt)
    R_after: Optional[np.ndarray] = None  # R after change (abrupt)
    drift_rate: float = 0.0     # Rate of linear drift
    period: int = 100           # Period for periodic regime

    def get_noise(self, t: int) -> tuple[np.ndarray, np.ndarray]:
        """Return (Q_true, R_true) at timestep t."""
        if self.noise_regime == "stationary":
            return np.diag(self.Q_base), np.diag(self.R_base)

        elif self.noise_regime == "abrupt_change":
            if self.change_time is not None and t >= self.change_time:
                q = self.Q_after if self.Q_after is not None else self.Q_base
                r = self.R_after if self.R_after is not None else self.R_base
                return np.diag(q), np.diag(r)
            return np.diag(self.Q_base), np.diag(self.R_base)

        elif self.noise_regime == "slow_drift":
            scale = 1.0 + self.drift_rate * t
            return np.diag(self.Q_base * scale), np.diag(self.R_base * scale)

        elif self.noise_regime == "periodic":
            scale = 1.0 + 0.5 * np.sin(2 * np.pi * t / self.period)
            return np.diag(self.Q_base * scale), np.diag(self.R_base * scale)

        return np.diag(self.Q_base), np.diag(self.R_base)


class TaskSampler:
    """Samples tasks from a calibrated distribution for meta-RL training."""

    def __init__(self, config: Optional[dict] = None):
        pass