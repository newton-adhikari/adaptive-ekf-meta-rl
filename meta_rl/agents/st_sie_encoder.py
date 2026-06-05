"""
Short-Time Spectral Innovation Encoder (ST-SIE).

We extracts noise-diagnostic features from the STFT of the
innovation sequence under a local stationarity assumption.

Simple architecture:
  Innovation buffer → STFT → |STFT|² → Log spectrogram
  → 2D-CNN → Spectral tokens (K/V)
  → Cross-attention with filter state (Q) → Latent context z
"""
