"""
Recursive Least Squares (RLS) covariance estimation baseline.
Linear estimator with forgetting factor for online Q/R adaptation.
"""

import numpy as np


class RLSCovarianceAdapter:
    """RLS-based adaptive noise covariance estimation.

    Uses recursive least squares with exponential forgetting
    to track Q and R online.
    """

    def __init__(
        self,
    ):
        self.Q = Q_init.copy()
        self.R = R_init.copy()
        self.lam = forgetting_factor

        n_x = Q_init.shape[0]
        n_z = R_init.shape[0]

        # RLS state for R estimation
        self._P_rls_R = np.eye(n_z) * 100.0  # RLS covariance
        self._r_vec = self.R.diagonal().copy()  # vectorized R diagonal

        # RLS state for Q estimation
        self._P_rls_Q = np.eye(n_x) * 100.0
        self._q_vec = self.Q.diagonal().copy()

        self._step = 0

    def adapt(
        self,
        innovation: np.ndarray,
        P: np.ndarray,
        S: np.ndarray,
        H: np.ndarray = None,
        K: np.ndarray = None,
        **kwargs,
    ) -> tuple[np.ndarray, np.ndarray]:
        self._step += 1

        # --- R estimation via RLS ---
        # Target: diagonal of (ν ν^T - H P H^T)
        nu_sq = innovation ** 2
        if H is not None:
            hph = np.diag(H @ P @ H.T)
            r_target = nu_sq - hph
        else:
            r_target = nu_sq - np.diag(S) + self._r_vec

        # RLS update for R diagonal
        phi_R = np.eye(len(self._r_vec))  # identity regressor
        for i in range(len(self._r_vec)):
            e_i = np.zeros(len(self._r_vec))
            e_i[i] = 1.0
            k_rls = (self._P_rls_R @ e_i) / (
                self.lam + e_i @ self._P_rls_R @ e_i
            )
            self._r_vec[i] += k_rls[i] * (r_target[i] - self._r_vec[i])
            self._P_rls_R = (
                self._P_rls_R - np.outer(k_rls, e_i @ self._P_rls_R)
            ) / self.lam

        # Ensure positive
        self._r_vec = np.maximum(self._r_vec, 1e-6)
        self.R = np.diag(self._r_vec)

        # --- Q estimation via RLS ---
        if K is not None:
            residual = K @ innovation
            q_target = residual ** 2

            for i in range(len(self._q_vec)):
                e_i = np.zeros(len(self._q_vec))
                e_i[i] = 1.0
                k_rls = (self._P_rls_Q @ e_i) / (
                    self.lam + e_i @ self._P_rls_Q @ e_i
                )
                self._q_vec[i] += k_rls[i] * (q_target[i] - self._q_vec[i])
                self._P_rls_Q = (
                    self._P_rls_Q - np.outer(k_rls, e_i @ self._P_rls_Q)
                ) / self.lam

            self._q_vec = np.maximum(self._q_vec, 1e-6)
            self.Q = np.diag(self._q_vec)

        return self.Q, self.R

    def reset(self):
        self._step = 0
        n_x = self.Q.shape[0]
        n_z = self.R.shape[0]
        self._P_rls_R = np.eye(n_z) * 100.0
        self._P_rls_Q = np.eye(n_x) * 100.0
