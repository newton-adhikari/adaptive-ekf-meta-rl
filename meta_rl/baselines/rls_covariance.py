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
        