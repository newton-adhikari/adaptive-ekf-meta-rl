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

    def __init__(self, filter_state_dim: int, embed_dim: int):
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
        self,
        innovation_dim: int = 2,
        filter_state_dim: int = 5,  # NEES + NIS + tr(P) + diag(S)[2]
        window_size: int = 32,
        hop_size: int = 8,
        latent_dim: int = 32,
        n_heads: int = 4,
        cnn_hidden: int = 32,
    ):
        super().__init__()
        self.innovation_dim = innovation_dim
        self.window_size = window_size
        self.hop_size = hop_size
        self.latent_dim = latent_dim

        n_freq = window_size // 2 + 1
        self.n_freq = n_freq

        # 2D-CNN over spectrogram
        self.spectral_cnn = SpectralCNN(
            in_channels=innovation_dim, hidden_dim=cnn_hidden
        )
        cnn_out_dim = self.spectral_cnn.out_dim

        # Project CNN output to attention dimension
        self.embed_dim = 64
        self.spectral_proj = nn.Linear(cnn_out_dim, self.embed_dim)

        # Filter state encoder (query)
        self.filter_encoder = FilterStateEncoder(filter_state_dim, self.embed_dim)

        # Cross-attention: filter state queries spectral tokens
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=self.embed_dim, num_heads=n_heads, batch_first=True
        )

        # Final projection to latent context
        self.output_proj = nn.Sequential(
            nn.Linear(self.embed_dim + filter_state_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, latent_dim),
        )

        # Hann window (registered as buffer, not parameter)
        self.register_buffer(
            "hann_window", torch.hann_window(window_size).float()
        )

    def compute_spectrogram(
        self, innovation_buffer: torch.Tensor
    ) -> torch.Tensor:
        """
        This computes log-power spectrogram from innovation buffer.
        """
        B, L, D = innovation_buffer.shape
        n_frames = max(0, (L - self.window_size) // self.hop_size + 1)

        if n_frames == 0:
            return torch.zeros(
                B, D, 1, self.n_freq, device=innovation_buffer.device
            )

        # Extract overlapping frames
        frames = []
        for i in range(n_frames):
            start = i * self.hop_size
            frame = innovation_buffer[:, start : start + self.window_size, :]
            frames.append(frame)

        # (B, n_frames, W, D)
        frames = torch.stack(frames, dim=1)

        # Apply Hann window
        window = self.hann_window[None, None, :, None]  # (1, 1, W, 1)
        frames = frames * window

        # FFT per frame per dimension: reshape to (B*n_frames*D, W)
        frames_flat = frames.permute(0, 1, 3, 2).reshape(-1, self.window_size)
        fft_result = torch.fft.rfft(frames_flat, dim=-1)
        power = (fft_result.abs() ** 2) / self.window_size

        # Reshape back: (B, n_frames, D, n_freq) → (B, D, n_frames, n_freq)
        power = power.reshape(B, n_frames, D, self.n_freq).permute(0, 2, 1, 3)

        # Log scale
        return torch.log(power + 1e-10)

    def forward(
        self,
        innovation_buffer: torch.Tensor,
        filter_state: torch.Tensor,
    ) -> torch.Tensor:
        """
        This encodes innovation history + filter state into latent context z.
        """
        # Compute spectrogram
        spectrogram = self.compute_spectrogram(innovation_buffer)

        # 2D-CNN features
        cnn_features = self.spectral_cnn(spectrogram)  # (B, cnn_out_dim)

        # Project to spectral tokens: (B, 1, embed_dim)
        spectral_tokens = self.spectral_proj(cnn_features).unsqueeze(1)

        # Filter state as query: (B, 1, embed_dim)
        query = self.filter_encoder(filter_state)

        # Cross-attention: filter asks spectrum what's wrong
        attended, _ = self.cross_attention(
            query=query, key=spectral_tokens, value=spectral_tokens
        )  # (B, 1, embed_dim)

        # Combine attended features with raw filter state
        combined = torch.cat([attended.squeeze(1), filter_state], dim=-1)
        z = self.output_proj(combined)

        return z
