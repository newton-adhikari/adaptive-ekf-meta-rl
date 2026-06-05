"""
This is raw GRU-based innovation encoder (baseline for ablation).

This is the "naive PEARL" encoder that processes raw innovations
with a GRU, without any spectral inductive bias.

This will be used to isolate the contribution of ST-SIE.
"""

import torch
import torch.nn as nn


class RawInnovationEncoder(nn.Module):
    """GRU encoder over raw innovation sequence (PEARL-style).

    No spectral processing, we use as ablation baseline for ST-SIE.

    """

    def __init__(
        self
    ):
        super().__init__()
        