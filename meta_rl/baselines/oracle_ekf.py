"""
Oracle EKF baseline: uses true Q/R at every timestep.
Performance ceiling — I think no real method can beat this.
"""

import numpy as np
from typing import Callable, Optional


class OracleEKFAdapter:
    """Oracle adapter that uses the true noise covariances.

    Requires access to the ground-truth noise schedule.
    Used as an upper bound on achievable performance.
    """

    def __init__(
        self,
        noise_schedule: Optional[Callable[[int], tuple[np.ndarray, np.ndarray]]] = None,
    ):
        """
        Args:
            noise_schedule: Function mapping timestep → (Q_true, R_true).
        """
        self.noise_schedule = noise_schedule
        self._step = 0
        self.Q = None
        self.R = None

    def set_noise_schedule(
        self, schedule: Callable[[int], tuple[np.ndarray, np.ndarray]]
    ):
        self.noise_schedule = schedule

    def adapt(
        self,
        innovation: np.ndarray,
        P: np.ndarray,
        S: np.ndarray,
        **kwargs,
    ) -> tuple[np.ndarray, np.ndarray]:
        if self.noise_schedule is not None:
            self.Q, self.R = self.noise_schedule(self._step)
        self._step += 1
        return self.Q, self.R

    def reset(self):
        self._step = 0
