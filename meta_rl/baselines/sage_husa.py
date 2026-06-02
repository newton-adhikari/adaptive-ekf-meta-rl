"""
This is Sage-Husa adaptive EKF baseline.
Exponential moving average of innovation outer product to estimate R,
and residual-based estimation for Q.
"""

class SageHusaAdapter:
    """Sage-Husa (1969) adaptive noise estimation.

    This assumes stationary noisem diverges under abrupt changes.
    """

    def __init__(
        self,
        Q_init: np.ndarray,
        R_init: np.ndarray,
        forgetting_factor: float = 0.98,
    ):
        self.Q = Q_init.copy()
        self.R = R_init.copy()
        self.b = forgetting_factor
        self._step = 0

    def adapt(
        self,
        innovation: np.ndarray,
        P: np.ndarray,
        S: np.ndarray,
        H: np.ndarray = None,
        F: np.ndarray = None,
        K: np.ndarray = None,
        x_pred: np.ndarray = None,
        x_updated: np.ndarray = None,
        **kwargs,
    ) -> tuple[np.ndarray, np.ndarray]:
        self._step += 1
        d_k = 1.0 - self.b if self._step > 1 else 1.0

        # for R estimation: R_k = (1-d_k)*R_{k-1} + d_k*(ν_k ν_k^T - H P H^T)
        nu_outer = np.outer(innovation, innovation)
        if H is not None:
            R_innov = nu_outer - H @ P @ H.T
        else:
            R_innov = nu_outer - S + self.R
        self.R = (1 - d_k) * self.R + d_k * R_innov

        # to ensure R stays positive definite
        self.R = np.maximum(self.R, np.eye(self.R.shape[0]) * 1e-6)

        # for Q estimation via residual
        if K is not None and F is not None:
            residual = K @ innovation
            q_outer = np.outer(residual, residual)
            self.Q = (1 - d_k) * self.Q + d_k * q_outer
            self.Q = np.maximum(self.Q, np.eye(self.Q.shape[0]) * 1e-6)

        return self.Q, self.R

    def reset(self):
        self._step = 0
