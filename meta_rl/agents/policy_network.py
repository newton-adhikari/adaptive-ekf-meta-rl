"""
Will create gaussian policy network for Q/R adjustment.

Outputs multiplicative scaling factors for EKF noise covariance diagonals.
Action space: [αQ_1, ..., αQ_nq, αR_1, ..., αR_nr] ∈ [0.1, 10.0].
"""

LOG_STD_MIN = -5.0
LOG_STD_MAX = 2.0