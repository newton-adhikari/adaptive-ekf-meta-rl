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
        