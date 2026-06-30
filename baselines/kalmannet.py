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

 