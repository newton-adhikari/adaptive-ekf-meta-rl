"""
Fixed EKF baseline: no adaptation, uses nominal Q/R throughout.
Lower bound on performance.
"""

import numpy as np


class FixedEKFAdapter:
    """Returns nominal Q/R without any adaptation."""

    def __init__(self, Q_nominal: np.ndarray, R_nominal: np.ndarray):
        self.Q = Q_nominal.copy()
        self.R = R_nominal.copy()