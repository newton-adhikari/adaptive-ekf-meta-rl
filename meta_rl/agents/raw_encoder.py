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
        self.innovation_dim = innovation_dim
        self.latent_dim = latent_dim

        # GRU over raw innovations
        self.gru = nn.GRU(
            input_size=innovation_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
        )

        # Combine GRU output with filter state
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim + filter_state_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, latent_dim),
        )

    def forward(
        self,
        innovation_buffer: torch.Tensor,
        filter_state: torch.Tensor,
    ) -> torch.Tensor:
        """Encode raw innovations + filter state into latent context.

        Returns:
            z: (B, latent_dim) latent context.
        """
        # GRU encoding — use last hidden state
        _, h_n = self.gru(innovation_buffer)  # h_n: (n_layers, B, hidden)
        gru_out = h_n[-1]  # (B, hidden)

        combined = torch.cat([gru_out, filter_state], dim=-1)
        return self.output_proj(combined)
