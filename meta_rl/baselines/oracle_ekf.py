"""
Oracle EKF baseline: uses true Q/R at every timestep.
Performance ceiling — I think no real method can beat this.
"""

import numpy as np
from typing import Callable, Optional


class OracleEKFAdapter:
    """Oracle adapter that uses the true noise covariances.

    Requires access to the ground-truth noise schedule.
    Used as an upper bound on achievable performance.
    """

    def __init__(
        self
    ):