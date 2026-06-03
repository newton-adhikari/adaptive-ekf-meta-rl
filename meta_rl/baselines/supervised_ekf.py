"""
Supervised NN baseline for EKF noise estimation.
This will train a neural network to predict Q_true/R_true from innovation windows.
Tests whether RL is necessary vs supervised learning.
"""

import numpy as np

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class SupervisedNoisePredictor(nn.Module):
    """MLP that predicts Q/R diagonal from innovation window."""

    def __init__(
        self,
        innovation_dim: int = 2,
        window_size: int = 50,
        n_q: int = 6,
        n_r: int = 2,
        hidden_dim: int = 128,
    ):
        super().__init__()
        input_dim = innovation_dim * window_size
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_q + n_r),
            nn.Softplus(),  # ensure positive outputs
        )

    def forward(self, innovation_window: torch.Tensor) -> torch.Tensor:
        """
        Args:
            innovation_window: (B, W, D) innovation buffer.
        Returns:
            params: (B, n_q + n_r) predicted Q/R diagonal elements.
        """
        x = innovation_window.flatten(start_dim=1)
        return self.net(x)

