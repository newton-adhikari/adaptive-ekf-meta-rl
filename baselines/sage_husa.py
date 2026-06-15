"""
This is Sage-Husa adaptive EKF baseline.
Exponential moving average of innovation outer product to estimate R,
and residual-based estimation for Q.
"""

import numpy as np


class SageHusaAdapter:
    """Sage-Husa (1969) adaptive noise estimation.

    Fix: We estimates R online from innovation statistics.
    Q is kept at nominal (not adapted) for stability.

    """

    def __init__(
        self,
        Q_init: np.ndarray,
        R_init: np.ndarray,
        forgetting_factor: float = 0.995,
        min_eigenvalue: float = 1e-4,
    ):
        # here we, have kept fixed Q — joint adaptation diverges
        self.Q = Q_init.copy()
        self.R = R_init.copy()
        self.b = forgetting_factor
        self.min_ev = min_eigenvalue
        self._step = 0

    def adapt(
        self,
        innovation: np.ndarray,
        P: np.ndarray,
        H: np.ndarray,
        **kwargs,
    ) -> tuple[np.ndarray, np.ndarray]:
        self._step += 1

        # Learning rate: transitions from averaging (1/step) to EMA (1-b)
        # Step 1: dk=1.0 (full weight to first observation)
        # Step 2: dk=0.5
        # ...
        # Step 200+: dk=0.005 (steady-state exponential)
        dk = min(1.0 - self.b, 1.0 / self._step)

        # E[ν*νᵀ] = H*P_k|k-1*Hᵀ + R  →  R ≈ ν*νᵀ - H*P*Hᵀ
        # storing the predicted covariance separately.
        R_innov = np.outer(innovation, innovation) - H @ P @ H.T

        # Exponential moving average update
        R_new = (1 - dk) * self.R + dk * R_innov

        ev, evec = np.linalg.eigh(R_new)
        ev = np.maximum(ev, self.min_ev)
        self.R = evec @ np.diag(ev) @ evec.T

        return self.Q, self.R

    def reset(self):
        """Reset step counter (e.g., at episode boundaries)."""
        self._step = 0

    def get_params(self) -> dict:
        """Return current noise estimates for logging."""
        return {
            "Q_diag": np.diag(self.Q).tolist(),
            "R_diag": np.diag(self.R).tolist(),
            "step": self._step,
            "dk": min(1.0 - self.b, 1.0 / max(self._step, 1)),
        }
 