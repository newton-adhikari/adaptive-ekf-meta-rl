"""
KalmanNet Baseline Implementation.

Simplified implementation of KalmanNet (Revach et al., 2022).
Replaces the Kalman gain computation with a learned GRU that predicts K
from the innovation sequence.

Key differences from CC-MetaEKF:
- KalmanNet: learns K directly (black-box gain, no explicit Q/R)
- CC-MetaEKF: learns Q/R adjustments (preserves EKF structure)

Training: Supervised with oracle Kalman gain as target.

Note on NEES evaluation:
  KalmanNet's predicted K is not derived from its maintained P. We propagate
  P using Q_nom and update with the learned K for NEES computation, but this
  P may not be statistically meaningful since K and P are decoupled. This is
  precisely the limitation our paper discusses: replacing explicit Q/R with a
  learned gain sacrifices the consistency analysis framework.

Reference:
  Revach et al., "KalmanNet: Neural Network Aided Kalman Filtering
  for Partially Known Dynamics," IEEE TSP, 2022.

"""

import numpy as np
import torch
import torch.nn as nn
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


class KalmanNetGRU(nn.Module):
    """KalmanNet: GRU-based Kalman gain predictor.

    Input: innovation + filter state diagnostics (per timestep)
    Output: Kalman gain matrix K (n × m)
    """

    def __init__(self, state_dim=6, meas_dim=2, hidden_dim=64, n_layers=2):
        super().__init__()
        self.state_dim = state_dim
        self.meas_dim = meas_dim
        self.gain_dim = state_dim * meas_dim  # K is n×m = 12

        # Input: innovation (m) + log(diag(P)) (n) + log(diag(S)) (m)
        input_dim = meas_dim + state_dim + meas_dim  # 2 + 6 + 2 = 10

        self.gru = nn.GRU(input_dim, hidden_dim, n_layers, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.gain_dim),
        )
        self.hidden = None

    def reset(self):
        """Reset GRU hidden state (call at episode boundaries)."""
        self.hidden = None

    def forward(self, innovation, P_diag, S_diag):
        """Predict Kalman gain from current innovation and filter state.

        Args:
            innovation: (batch, m) current innovation vector
            P_diag: (batch, n) log1p(diag(P)) — predicted covariance
            S_diag: (batch, m) log1p(diag(S)) — innovation covariance
        Returns:
            K: (batch, n, m) predicted Kalman gain
        """
        x = torch.cat([innovation, P_diag, S_diag], dim=-1)
        x = x.unsqueeze(1)  # (batch, 1, input_dim) — single timestep

        # Detach hidden to prevent backprop through previous optimizer steps
        if self.hidden is not None:
            self.hidden = self.hidden.detach()

        out, self.hidden = self.gru(x, self.hidden)
        K_flat = self.fc(out.squeeze(1))
        K = K_flat.reshape(-1, self.state_dim, self.meas_dim)
        return K


# ================================================================
# Main
# ================================================================

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="KalmanNet baseline")
    parser.add_argument("--train", action="store_true", help="Train KalmanNet")
    parser.add_argument("--eval-kitti", action="store_true", help="Evaluate on KITTI")
    parser.add_argument("--episodes", type=int, default=5000)
    parser.add_argument("--checkpoint", type=str, default="checkpoints/kalmannet.pt")
    args = parser.parse_args()

 