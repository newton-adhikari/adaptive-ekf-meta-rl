"""
Variational Bayes EKF (VB-EKF) baseline.
Classical adaptive method using VB-EM updates for Q/R.
Based on Sarkka & Nummenmaa (2009).
"""


class VBEKFAdapter:
    """Variational Bayes adaptive EKF.

    Uses inverse-Wishart priors on Q and R, updated via
    variational EM at each timestep.
    """

    def __init__(
        self
    ):
        self.n_x = Q_init.shape[0]
        self.n_z = R_init.shape[0]
        self.n_vb_iters = n_vb_iters
        self.rho = rho  # forgetting factor

        # Inverse-Wishart parameters for R
        self.tau_R = float(self.n_z + 2)  # degrees of freedom
        self.T_R = R_init * (self.tau_R - self.n_z - 1)  # scale matrix

        # Inverse-Wishart parameters for Q
        self.tau_Q = float(self.n_x + 2)
        self.T_Q = Q_init * (self.tau_Q - self.n_x - 1)

        self.Q = Q_init.copy()
        self.R = R_init.copy()

    def adapt(
        self,
        innovation: np.ndarray,
        P: np.ndarray,
        S: np.ndarray,
        H: np.ndarray = None,
        F: np.ndarray = None,
        x_pred: np.ndarray = None,
        x_updated: np.ndarray = None,
        P_pred: np.ndarray = None,
        **kwargs,
    ) -> tuple[np.ndarray, np.ndarray]:
        if H is None or F is None:
            return self.Q, self.R

        # VB-EM iterations
        R_est = self.R.copy()
        Q_est = self.Q.copy()

        for _ in range(self.n_vb_iters):
            # E-step: update state estimate with current R, Q
            S_vb = H @ P @ H.T + R_est
            K_vb = P @ H.T @ np.linalg.inv(S_vb)

            # M-step: update R
            nu_outer = np.outer(innovation, innovation)
            B_R = nu_outer + H @ P @ H.T
            self.tau_R = self.rho * self.tau_R + 1
            self.T_R = self.rho * self.T_R + B_R
            R_est = self.T_R / (self.tau_R - self.n_z - 1)

            # M-step: update Q (if we have prediction info)
            if P_pred is not None and x_pred is not None and x_updated is not None:
                dx = x_updated - F @ x_pred if F is not None else np.zeros(self.n_x)
                B_Q = np.outer(dx, dx) + P - F @ P_pred @ F.T if F is not None else np.eye(self.n_x) * 0.01
                self.tau_Q = self.rho * self.tau_Q + 1
                self.T_Q = self.rho * self.T_Q + B_Q
                Q_est = self.T_Q / (self.tau_Q - self.n_x - 1)

        # Ensure positive definite
        self.R = self._ensure_pd(R_est)
        self.Q = self._ensure_pd(Q_est)

        return self.Q, self.R

    @staticmethod
    def _ensure_pd(M: np.ndarray, eps: float = 1e-6) -> np.ndarray:
        eigvals, eigvecs = np.linalg.eigh(M)
        eigvals = np.maximum(eigvals, eps)
        return eigvecs @ np.diag(eigvals) @ eigvecs.T

    def reset(self):
        self.tau_R = float(self.n_z + 2)
        self.tau_Q = float(self.n_x + 2)
