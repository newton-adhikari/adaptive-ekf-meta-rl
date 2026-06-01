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
        config = config or {}
        self.state_dim = config.get("state_dim", 6)
        self.meas_dim = config.get("meas_dim", 2)

        # Q/R ranges (log-uniform sampling)
        self.q_range = config.get("q_range", (0.01, 1.0))
        self.r_range = config.get("r_range", (0.1, 10.0))

        # Noise regime probabilities
        self.regime_probs = config.get("regime_probs", {
            "stationary": 0.25,
            "abrupt_change": 0.35,
            "slow_drift": 0.20,
            "periodic": 0.20,
        })

        self.trajectory_types = config.get(
            "trajectory_types", ["circle", "figure8", "straight", "random_walk"]
        )

        # Abrupt change parameters
        self.change_time_range = config.get("change_time_range", (50, 150))
        self.change_factor_range = config.get("change_factor_range", (2.0, 10.0))

    def sample(self, rng: Optional[np.random.Generator] = None) -> TaskConfig:
        """Sample a random task from the distribution."""
        rng = rng or np.random.default_rng()

        # Sample base Q/R (log-uniform)
        Q_base = np.exp(rng.uniform(
            np.log(self.q_range[0]), np.log(self.q_range[1]), self.state_dim
        ))
        R_base = np.exp(rng.uniform(
            np.log(self.r_range[0]), np.log(self.r_range[1]), self.meas_dim
        ))

        # Sample noise regime
        regimes = list(self.regime_probs.keys())
        probs = list(self.regime_probs.values())
        regime = rng.choice(regimes, p=probs)

        # Sample trajectory type
        traj_type = rng.choice(self.trajectory_types)

        # Build task config
        task = TaskConfig(
            Q_base=Q_base,
            R_base=R_base,
            noise_regime=regime,
            trajectory_type=traj_type,
        )

        if regime == "abrupt_change":
            task.change_time = int(rng.integers(*self.change_time_range))
            factor = rng.uniform(*self.change_factor_range)
            # Randomly scale up or down
            if rng.random() > 0.5:
                task.Q_after = Q_base * factor
                task.R_after = R_base.copy()
            else:
                task.Q_after = Q_base.copy()
                task.R_after = R_base * factor

        elif regime == "slow_drift":
            task.drift_rate = rng.uniform(0.001, 0.01)

        elif regime == "periodic":
            task.period = int(rng.integers(50, 200))

        return task

    def sample_batch(
        self, n: int, rng: Optional[np.random.Generator] = None
    ) -> list[TaskConfig]:
        """Sample a batch of tasks."""
        rng = rng or np.random.default_rng()
        return [self.sample(rng) for _ in range(n)]
