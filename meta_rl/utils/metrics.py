"""
Metrics for EKF evaluation: NEES, NIS, RMSE, consistency rate.
"""

import numpy as np
from scipy.stats import chi2


def compute_nees(x_true: np.ndarray, x_est: np.ndarray, P: np.ndarray) -> float:
    """Normalized Estimation Error Squared.

    NEES = (x_true - x_est)^T P^{-1} (x_true - x_est)
    For a consistent filter, E[NEES] = n_x (state dimension).
    """
    e = x_true - x_est
    # Regularize near-singular covariance
    P_reg = P + np.eye(P.shape[0]) * 1e-8
    try:
        return float(e.T @ np.linalg.inv(P_reg) @ e)
    except np.linalg.LinAlgError:
        return float('nan')


def compute_nis(innovation: np.ndarray, S: np.ndarray) -> float:
    """Normalized Innovation Squared.

    NIS = nu^T S^{-1} nu
    For a consistent filter, E[NIS] = n_z (measurement dimension).
    """
    return float(innovation.T @ np.linalg.inv(S) @ innovation)


def chi2_bounds(dim: int, confidence: float = 0.95) -> tuple[float, float]:
    """Chi-squared confidence interval bounds."""
    alpha = 1.0 - confidence
    lb = chi2.ppf(alpha / 2, df=dim)
    ub = chi2.ppf(1 - alpha / 2, df=dim)
    return lb, ub


def consistency_rate(
    nees_values: np.ndarray, state_dim: int, confidence: float = 0.95
) -> float:
    """Fraction of timesteps where NEES is within chi-squared bounds."""
    lb, ub = chi2_bounds(state_dim, confidence)
    within = np.logical_and(nees_values >= lb, nees_values <= ub)
    return float(np.mean(within))


def compute_rmse(x_true: np.ndarray, x_est: np.ndarray) -> float:
    """Root Mean Squared Error across a trajectory.

    Args:
        x_true: (T, n_x) ground truth states.
        x_est: (T, n_x) estimated states.
    """
    return float(np.sqrt(np.mean((x_true - x_est) ** 2)))


def compute_position_rmse(x_true: np.ndarray, x_est: np.ndarray) -> float:
    """RMSE on position components only (first 2 dims)."""
    return float(np.sqrt(np.mean((x_true[:, :2] - x_est[:, :2]) ** 2)))


def compute_heading_rmse(x_true: np.ndarray, x_est: np.ndarray) -> float:
    """RMSE on heading (3rd dim), handling angle wrapping."""
    diff = x_true[:, 2] - x_est[:, 2]
    diff = (diff + np.pi) % (2 * np.pi) - np.pi
    return float(np.sqrt(np.mean(diff ** 2)))


def average_nees(nees_values: np.ndarray) -> float:
    """Time-averaged NEES (ANEES)."""
    return float(np.mean(nees_values))


def compute_all_metrics(
    x_true_traj: np.ndarray,
    x_est_traj: np.ndarray,
    P_traj: np.ndarray,
    innovation_traj: np.ndarray,
    S_traj: np.ndarray,
    state_dim: int,
    meas_dim: int,
) -> dict:
    """Compute all metrics over a trajectory.

    Args:
        x_true_traj: (T, n_x)
        x_est_traj: (T, n_x)
        P_traj: (T, n_x, n_x)
        innovation_traj: (T, n_z)
        S_traj: (T, n_z, n_z)
    """
    T = x_true_traj.shape[0]
    nees_vals = np.array([
        compute_nees(x_true_traj[t], x_est_traj[t], P_traj[t]) for t in range(T)
    ])
    nis_vals = np.array([
        compute_nis(innovation_traj[t], S_traj[t]) for t in range(T)
    ])

    return {
        "nees_values": nees_vals,
        "nis_values": nis_vals,
        "anees": average_nees(nees_vals),
        "consistency_rate": consistency_rate(nees_vals, state_dim),
        "rmse": compute_rmse(x_true_traj, x_est_traj),
        "position_rmse": compute_position_rmse(x_true_traj, x_est_traj),
        "heading_rmse": compute_heading_rmse(x_true_traj, x_est_traj),
    }
