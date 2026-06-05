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
        
