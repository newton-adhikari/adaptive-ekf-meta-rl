"""
Will create gaussian policy network for Q/R adjustment.

Outputs multiplicative scaling factors for EKF noise covariance diagonals.
Action space: [αQ_1, ..., αQ_nq, αR_1, ..., αR_nr] ∈ [0.1, 10.0].
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

LOG_STD_MIN = -5.0
LOG_STD_MAX = 2.0



class GaussianPolicy(nn.Module):
    """
    This is squashed Gaussian policy for SAC-based Q/R adaptation.

    """

    def __init__(
        self
    ):
        super().__init__()