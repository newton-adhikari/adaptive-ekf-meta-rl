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
        self
    ):
        pass