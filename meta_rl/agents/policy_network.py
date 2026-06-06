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
        self,
        state_dim: int,
        context_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        action_low: float = 0.1,
        action_high: float = 10.0,
    ):
        super().__init__()
        self.action_low = action_low
        self.action_high = action_high

        input_dim = state_dim + context_dim

        self.trunk = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.mean_head = nn.Linear(hidden_dim, action_dim)
        self.log_std_head = nn.Linear(hidden_dim, action_dim)

    def forward(
        self, state: torch.Tensor, context: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute mean and log_std of Gaussian policy.

        """
        x = torch.cat([state, context], dim=-1)
        h = self.trunk(x)
        mean = self.mean_head(h)
        log_std = self.log_std_head(h)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        return mean, log_std

    def sample(
        self, state: torch.Tensor, context: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Sample action with reparameterization trick + squashing.

        """
        mean, log_std = self.forward(state, context)
        std = log_std.exp()
        dist = Normal(mean, std)

        # Reparameterized sample
        x_t = dist.rsample()

        # Squash to [0, 1] via sigmoid, then scale to [low, high]
        y_t = torch.sigmoid(x_t)
        action = self.action_low + (self.action_high - self.action_low) * y_t

        # Log probability with squashing correction
        log_prob = dist.log_prob(x_t)

        # Jacobian correction for sigmoid squashing
        log_prob -= torch.log(
            (self.action_high - self.action_low) * y_t * (1 - y_t) + 1e-6
        )
        log_prob = log_prob.sum(dim=-1, keepdim=True)

        return action, log_prob

    def deterministic(
        self, state: torch.Tensor, context: torch.Tensor
    ) -> torch.Tensor:
        """
        This is the deterministic action (mean, squashed)  which is for evaluation."""
        mean, _ = self.forward(state, context)
        y = torch.sigmoid(mean)
        return self.action_low + (self.action_high - self.action_low) * y


class QNetwork(nn.Module):
    """Q-value network for SAC critic.

    Args:
        state_dim: Observation dimension.
        context_dim: Latent context dimension.
        action_dim: Action dimension.
        hidden_dim: Hidden layer size.
    """

    def __init__(
        self,
        state_dim: int,
        context_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
    ):
        super().__init__()
        input_dim = state_dim + context_dim + action_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        context: torch.Tensor,
    ) -> torch.Tensor:
        x = torch.cat([state, action, context], dim=-1)
        return self.net(x)
