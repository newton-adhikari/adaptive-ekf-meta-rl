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
        
    ):
        pass