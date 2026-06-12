"""
Sensor characterization: Allan variance analysis, noise model fitting.

This will be used to build calibrated task distributions from real sensor data.
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass
class AllanVarianceResult:
    """Results from Allan variance analysis."""
    taus: np.ndarray           # averaging times
    adev: np.ndarray           # Allan deviation values
    arw: float                 # angle random walk (slope = -0.5)
    bias_instability: float    # bias instability (minimum of adev)
    rrw: float                 # rate random walk (slope = +0.5)



def compute_allan_variance(
    data: np.ndarray, sample_rate: float, max_clusters: int = 100
) -> AllanVarianceResult:
    """
    Method for computing Allan variance for IMU characterization.
    with taus, adev, and fitted noise parameters.
    """
    N = len(data)
    dt = 1.0 / sample_rate

    # Compute cumulative sum (integration)
    theta = np.cumsum(data) * dt

    # Cluster sizes (log-spaced)
    max_m = N // 2
    m_values = np.unique(
        np.logspace(0, np.log10(max_m), max_clusters).astype(int)
    )
    m_values = m_values[m_values > 0]

    taus = m_values * dt
    adev = np.zeros(len(m_values))

    for i, m in enumerate(m_values):
        # Allan variance: σ²(τ) = 1/(2τ²(N-2m)) Σ(θ[k+2m] - 2θ[k+m] + θ[k])²
        tau = m * dt
        diffs = theta[2 * m :] - 2 * theta[m : -m] + theta[: -2 * m]
        adev[i] = np.sqrt(np.mean(diffs ** 2) / (2 * tau ** 2))

    # Fit noise parameters from log-log slope
    log_tau = np.log10(taus)
    log_adev = np.log10(adev + 1e-15)

    # ARW: slope = -0.5 region (short tau)
    short_mask = taus < np.median(taus) * 0.1
    if np.sum(short_mask) >= 2:
        arw = float(10 ** np.interp(0, log_tau[short_mask], log_adev[short_mask]))
    else:
        arw = float(adev[0] * np.sqrt(taus[0]))

    # Bias instability: minimum of adev
    bias_instability = float(np.min(adev))

    # RRW: slope = +0.5 region (long tau)
    long_mask = taus > np.median(taus) * 10
    if np.sum(long_mask) >= 2:
        rrw = float(adev[long_mask][-1] / np.sqrt(taus[long_mask][-1]))
    else:
        rrw = float(adev[-1] / np.sqrt(taus[-1]))

    return AllanVarianceResult(
        taus=taus, adev=adev, arw=arw,
        bias_instability=bias_instability, rrw=rrw,
    )

