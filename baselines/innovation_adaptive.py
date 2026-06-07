"""
Innovation-based adaptive EKF baseline.
Estimates R from windowed covariance: R = (1/W) Σ ν_t ν_t^T - H P H^T.
This is simpler alternative to Sage-Husa.
"""

import numpy as np
from collections import deque


class InnovationAdaptiveAdapter:
    """Innovation-based covariance estimation (Mehra, 1970/1972).

    Uses a sliding window of innovations to estimate R.
    Q is kept fixed (or optionally adapted via a secondary estimator).
    """

    def __init__(
        self,
        Q_init: np.ndarray,
        R_init: np.ndarray,
        window_size: int = 50,
        adapt_Q: bool = False,
    ):
        self.Q = Q_init.copy()
        self.R = R_init.copy()
        self.window_size = window_size
        self.adapt_Q = adapt_Q
        self._innovations: deque = deque(maxlen=window_size)
        self._residuals: deque = deque(maxlen=window_size)

    def adapt(
        self,
        innovation: np.ndarray,
        P: np.ndarray,
        S: np.ndarray,
        H: np.ndarray = None,
        K: np.ndarray = None,
        **kwargs,
    ) -> tuple[np.ndarray, np.ndarray]:
        self._innovations.append(innovation.copy())

        if len(self._innovations) >= self.window_size:
            innovations = np.array(self._innovations)

            # R = (1/W) Σ ν_t ν_t^T - H P H^T
            C_nu = np.mean(
                [np.outer(v, v) for v in innovations], axis=0
            )
            if H is not None:
                self.R = C_nu - H @ P @ H.T
            else:
                self.R = C_nu - S + self.R

            # Ensure positive definite
            eigvals = np.linalg.eigvalsh(self.R)
            if np.any(eigvals <= 0):
                self.R = self.R + np.eye(self.R.shape[0]) * (
                    abs(min(eigvals.min(), 0)) + 1e-6
                )

        # Optional Q adaptation
        if self.adapt_Q and K is not None:
            residual = K @ innovation
            self._residuals.append(residual)
            if len(self._residuals) >= self.window_size:
                residuals = np.array(self._residuals)
                self.Q = np.mean(
                    [np.outer(r, r) for r in residuals], axis=0
                )
                eigvals = np.linalg.eigvalsh(self.Q)
                if np.any(eigvals <= 0):
                    self.Q = self.Q + np.eye(self.Q.shape[0]) * (
                        abs(min(eigvals.min(), 0)) + 1e-6
                    )

        return self.Q, self.R

    def reset(self):
        self._innovations.clear()
        self._residuals.clear()
