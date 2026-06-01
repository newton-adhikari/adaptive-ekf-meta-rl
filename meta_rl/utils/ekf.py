#CC-MetaEKF: Filter-Aware Meta-RL for Online EKF Adaptation.
# main logic will be here


import numpy as np
from scipy.stats import chi2
from dataclasses import dataclass


@dataclass
class EKFState:
    x: np.ndarray       # state estimate [px, py, theta, vx, vy, omega]
    P: np.ndarray       # covariance matrix
    Q: np.ndarray       # process noise (adapted by policy)
    R: np.ndarray       # measurement noise (adapted by policy)
    innovation: np.ndarray  # last innovation
    S: np.ndarray       # innovation covariance
    nees: float         # normalized estimation error squared
    nis: float          # normalized innovation squared


class EKF:
    def __init__(self, state_dim=6, meas_dim=2, dt=0.1):
        self.n = state_dim
        self.m = meas_dim
        self.dt = dt

        # Chi-squared bounds for consistency (95% confidence)
        self.chi2_lb = chi2.ppf(0.025, df=state_dim)
        self.chi2_ub = chi2.ppf(0.975, df=state_dim)
        self.chi2_lb_nis = chi2.ppf(0.025, df=meas_dim)
        self.chi2_ub_nis = chi2.ppf(0.975, df=meas_dim)

    def reset(self, x0, P0, Q, R):
        self.state = EKFState(
            x=x0.copy(), P=P0.copy(), Q=Q.copy(), R=R.copy(),
            innovation=np.zeros(self.m), S=np.eye(self.m),
            nees=0.0, nis=0.0
        )
        self.innovation_history = []

    def set_noise(self, Q, R):
        self.state.Q = Q.copy()
        self.state.R = R.copy()

    def _motion_model(self, x, u):
        """Constant-velocity model with rotation."""
        px, py, theta, vx, vy, omega = x
        dt = self.dt
        return np.array([
            px + vx * np.cos(theta) * dt - vy * np.sin(theta) * dt,
            py + vx * np.sin(theta) * dt + vy * np.cos(theta) * dt,
            theta + omega * dt,
            vx + u[0] * dt,  # linear acceleration
            vy + u[1] * dt,
            omega + u[2] * dt,  # angular acceleration
        ])

    def _motion_jacobian(self, x, u):
        """Jacobian of motion model w.r.t. state."""
        _, _, theta, vx, vy, _ = x
        dt = self.dt
        F = np.eye(self.n)
        F[0, 2] = (-vx * np.sin(theta) - vy * np.cos(theta)) * dt
        F[0, 3] = np.cos(theta) * dt
        F[0, 4] = -np.sin(theta) * dt
        F[1, 2] = (vx * np.cos(theta) - vy * np.sin(theta)) * dt
        F[1, 3] = np.sin(theta) * dt
        F[1, 4] = np.cos(theta) * dt
        F[2, 5] = dt
        return F

    def _measurement_model(self, x):
        """Position measurement (e.g., from LiDAR scan matching)."""
        return x[:self.m]  # observe [px, py]

    def _measurement_jacobian(self, x):
        """Jacobian of measurement model."""
        H = np.zeros((self.m, self.n))
        H[:self.m, :self.m] = np.eye(self.m)
        return H

    def predict(self, u):
        x, P, Q = self.state.x, self.state.P, self.state.Q
        F = self._motion_jacobian(x, u)
        self.state.x = self._motion_model(x, u)
        self.state.P = F @ P @ F.T + Q

    def update(self, z):
        x, P, R = self.state.x, self.state.P, self.state.R
        H = self._measurement_jacobian(x)
        z_pred = self._measurement_model(x)

        # Innovation
        nu = z - z_pred
        S = H @ P @ H.T + R
        K = P @ H.T @ np.linalg.inv(S)

        # State update
        self.state.x = x + K @ nu
        self.state.P = (np.eye(self.n) - K @ H) @ P

        # Store diagnostics
        self.state.innovation = nu
        self.state.S = S
        self.state.nis = float(nu.T @ np.linalg.inv(S) @ nu)
        self.innovation_history.append(nu.copy())

        return nu

    def compute_nees(self, x_true):
        """Normalized Estimation Error Squared."""
        e = x_true - self.state.x
        P_inv = np.linalg.inv(self.state.P)
        self.state.nees = float(e.T @ P_inv @ e)
        return self.state.nees

    def is_consistent(self):
        return self.chi2_lb <= self.state.nees <= self.chi2_ub

    def get_innovation_window(self, window_size=32):
        """Return last W innovations, zero-padded if needed."""
        history = self.innovation_history[-window_size:]
        if len(history) < window_size:
            pad = [np.zeros(self.m)] * (window_size - len(history))
            history = pad + history
        return np.array(history)  # (W, m)

    @property
    def error(self):
        """Must be set externally: self._x_true"""
        return self._x_true - self.state.x if hasattr(self, '_x_true') else None
