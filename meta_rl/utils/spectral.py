"""
Spectral utilities: STFT, spectrogram computation, incremental sliding-window FFT.

Used by the ST-SIE encoder to extract noise-diagnostic features from
the innovation sequence under a local stationarity assumption.
"""

import numpy as np
from typing import Optional


def compute_stft(
    signal: np.ndarray,
    window_size: int = 32,
    hop_size: int = 8,
    window_fn: str = "hann",
) -> np.ndarray:
    """Short-Time Fourier Transform of a (possibly multi-dim) signal.
    """
    if signal.ndim == 1:
        signal = signal[:, None]

    T, D = signal.shape
    n_freq = window_size // 2 + 1

    # Window function
    if window_fn == "hann":
        window = np.hanning(window_size)
    elif window_fn == "hamming":
        window = np.hamming(window_size)
    else:
        window = np.ones(window_size)

    # Compute frames
    n_frames = max(0, (T - window_size) // hop_size + 1)
    stft_result = np.zeros((n_frames, n_freq, D), dtype=np.complex128)

    for i in range(n_frames):
        start = i * hop_size
        frame = signal[start : start + window_size] * window[:, None]
        stft_result[i] = np.fft.rfft(frame, axis=0)

    return stft_result


def compute_spectrogram(
    signal: np.ndarray,
    window_size: int = 32,
    hop_size: int = 8,
    window_fn: str = "hann",
    log_scale: bool = True,
    eps: float = 1e-10,
) -> np.ndarray:
    """Compute log power spectrogram from input signal.
    """
    stft = compute_stft(signal, window_size, hop_size, window_fn)
    power = np.abs(stft) ** 2 / window_size
    if log_scale:
        return np.log(power + eps)
    return power


def incremental_stft(
    buffer: np.ndarray,
    new_sample: np.ndarray,
    window_size: int = 32,
    window_fn: str = "hann",
) -> tuple[np.ndarray, np.ndarray]:
    """
    
    core logic for incremental sliding-window FFT for real-time operation.

    Appends new_sample to buffer, drops oldest if buffer exceeds window_size,
    and computes FFT of the current window. O(W log W) per step.
    """
    if new_sample.ndim == 0:
        new_sample = new_sample[None]

    buffer = np.vstack([buffer, new_sample[None, :]]) if buffer.size > 0 else new_sample[None, :]

    if buffer.shape[0] > window_size:
        buffer = buffer[-window_size:]

    if buffer.shape[0] < window_size:
        return buffer, None

    # Apply window and compute FFT
    if window_fn == "hann":
        window = np.hanning(window_size)
    elif window_fn == "hamming":
        window = np.hamming(window_size)
    else:
        window = np.ones(window_size)

    windowed = buffer * window[:, None]
    fft_result = np.fft.rfft(windowed, axis=0)
    return buffer, fft_result


def compute_psd(
    signal: np.ndarray, window_size: Optional[int] = None
) -> np.ndarray:
    """Power Spectral Density estimate using periodogram.
    """
    if signal.ndim == 1:
        signal = signal[:, None]
    if window_size is not None:
        signal = signal[-window_size:]
    T = signal.shape[0]
    fft_result = np.fft.rfft(signal, axis=0)
    return np.abs(fft_result) ** 2 / T
