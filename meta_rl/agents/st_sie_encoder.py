"""
Short-Time Spectral Innovation Encoder (ST-SIE).

We extracts noise-diagnostic features from the STFT of the
innovation sequence under a local stationarity assumption.

Simple architecture:
  Innovation buffer → STFT → |STFT|² → Log spectrogram
  → 2D-CNN → Spectral tokens (K/V)
  → Cross-attention with filter state (Q) → Latent context z
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional

class SpectralCNN(nn.Module):
    """This si a simple lightweight 2D-CNN over log-power spectrogram (time × frequency)."""

    def __init__(self, in_channels: int, hidden_dim: int = 32):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim * 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),  # fixed spatial output
        )
        self.out_dim = hidden_dim * 2 * 4 * 4

    def forward(self, spectrogram: torch.Tensor) -> torch.Tensor:
        """
        spectrogram: (B, C, n_frames, n_freq) log-power spectrogram.
        
        Returns:
            features: (B, out_dim) flattened CNN features.
        """
        x = self.conv(spectrogram)
        return x.flatten(start_dim=1)


class FilterStateEncoder(nn.Module):
    """This encodes filter diagnostics (NEES, NIS, tr(P), diag(S)) into tokens."""

    def __init__():
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(filter_state_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, filter_state: torch.Tensor) -> torch.Tensor:
        """
        Returns:
            tokens: (B, 1, embed_dim) — single query token.
        """
        return self.fc(filter_state).unsqueeze(1)


class STSIEEncoder(nn.Module):
    """Short-Time Spectral Innovation Encoder.

    Two loop design:
      - Slow loop: recompute context z every K EKF steps.
      - Fast loop: policy uses cached z at every EKF step.
    """

    def __init__(
        self
    ):
        super().__init__()
        