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
