# CC-MetaEKF: Constrained RL for Online EKF Noise Adaptation

Learns to adapt EKF process/measurement noise covariances online under non-stationary conditions, preserving the classical filter structure while achieving calibrated uncertainty.

**Key result:** 71.3% ± 6.2% NEES consistency on KITTI odometry (zero-shot transfer) vs. 55.9% for best fixed heuristic and 49.7% for Oracle EKF with true noise parameters.

## What This Does

The Extended Kalman Filter needs correct Q (process noise) and R (measurement noise) to produce calibrated uncertainty estimates. When conditions change at runtime — terrain transitions, sensor degradation, weather — fixed parameters fail silently.

CC-MetaEKF trains a policy via constrained PPO that observes the EKF's innovation sequence and outputs per-timestep Q/R scaling factors. Two components:

1. **ST-SIE** (Short-Time Spectral Innovation Encoder): Applies STFT to the innovation sequence, extracts noise-diagnostic features via 2D-CNN, fuses with filter state through cross-attention.

2. **PID-CCPO** (PID-Constrained Policy Optimization): Enforces NEES consistency as a constraint using PID-controlled Lagrangian relaxation, preventing the degenerate solution of covariance inflation.

## Results

| Method | Simulation Consistency | KITTI Transfer |
|--------|:---:|:---:|
| Fixed EKF | 50.6% | 36.6% |
| Sage-Husa | 50.1% | 37.4% |
| Oracle (true Q/R) | 56.4% | 49.7% |
| Best Q-inflation (Q×4) | — | 55.9% |
| **CC-MetaEKF (ours)** | **77.1%** | **71.3% ± 6.2%** |

Trained on simple 6-DOF unicycle trajectories (1 m/s, 10s episodes). Transfers zero-shot to KITTI vehicle trajectories (5-10 m/s, 110-466s, urban driving).

## Quick Start

### Install

```bash
git clone https://github.com/newton-adhikari/adaptive-ekf-meta-rl.git
cd adaptive-ekf-meta-rl
pip install -r requirements.txt
```

Requirements: Python 3.10+, PyTorch 2.0+, NumPy, SciPy.

### Train (full pipeline)

```bash
# Runs Phase 0 (1D feasibility) + Phase 1 (6D training) + Ablation + Comparison
python3 run_all.py --seed 42

# Single phase only
python3 run_all.py --phase 1 --seed 42          # ST-SIE + PID-CCPO only
python3 run_all.py --phase ablation --seed 42    # 4-variant ablation matrix
python3 run_all.py --phase comparison --seed 42  # Baseline comparison
```

Output: `results/run_s42/` with checkpoints, JSON results, and training logs.

### Evaluate on KITTI (no retraining needed)

```bash
# Single seed
python3 scripts/eval_kitti.py \
    --checkpoint results/run_s42/best_stsie_pid_s42.pt \
    --sequences 00 02 05 07

# Multi-seed with Q-inflation baselines
./scripts/run_kitti_multiseed.sh
```

Downloads KITTI ground truth poses automatically (~4MB). Results saved to `results/kitti_results.json`.

### Use a trained checkpoint

```python
import torch
import numpy as np
from run_all import EKF6D, STSIEEncoder, Policy

# Load
enc = STSIEEncoder(2, 4, 32, 16, 4)
policy = Policy(27, 8, enc)
policy.load_state_dict(torch.load("results/run_s42/best_stsie_pid_s42.pt", map_location="cpu"))
policy.eval()

# At each EKF step:
# obs = [last_5_innovations, nees/6, nis/2, log1p(trP), log1p(diagP), log1p(diagQ), log1p(diagR)]
# ib  = innovation_buffer (30, 2)
# fs  = [nees/6, nis/2, log1p(trP), log1p(S[0,0])]
with torch.no_grad():
    dist, _, _ = policy(obs_tensor, ib_tensor, fs_tensor)
    action = dist.mean.squeeze().numpy()

alphas = np.exp(np.clip(action, -2, 2))
Q_new = np.diag(alphas[:6]) @ Q_nominal
R_new = np.diag(alphas[6:]) @ R_nominal
```

## Project Structure

```
├── run_all.py                    # One-command training pipeline
├── meta_rl/
│   ├── agents/
│   │   ├── st_sie_encoder.py     # ST-SIE encoder
│   │   ├── policy_network.py     # Gaussian policy
│   │   └── pid_ccpo_agent.py     # PID-Lagrangian constrained agent
│   ├── envs/
│   │   └── lightweight_ekf_env.py # Gymnasium EKF environment
│   └── utils/
│       ├── ekf.py                # Standalone 6-DOF EKF
│       └── metrics.py            # NEES, consistency, RMSE
├── baselines/
│   ├── sage_husa.py              # Sage-Husa adaptive EKF
│   ├── innovation_adaptive.py    # Innovation-based method
│   ├── variational_bayes_ekf.py  # VB-EKF
│   └── oracle_ekf.py            # Oracle with true noise
├── scripts/
│   ├── eval_kitti.py             # KITTI odometry evaluation
│   └── run_kitti_multiseed.sh    # Multi-seed KITTI benchmark
├── ros2_ws/                      # ROS2 deployment nodes (TurtleBot4)
├── configs/                      # Training/eval YAML configs
├── paper/                        # LaTeX source
└── results/                      # Checkpoints + JSON results
```

## Reproducing Results

### Simulation experiments

```bash
# Train all 5 seeds (runs sequentially, ~55 hours total)
for seed in 42 123 456 789 1024; do
    python3 run_all.py --phase 1 --seed $seed
done

# Ablation (seed 42 only, ~11 hours)
python3 run_all.py --phase ablation --seed 42

# Baseline comparison
python3 run_all.py --phase comparison --seed 42
```

### KITTI transfer

```bash
# Requires trained checkpoints from above
./scripts/run_kitti_multiseed.sh
```

### Pre-trained checkpoints

Trained checkpoints are included in `results/run_s*/best_stsie_pid_s*.pt` for all 5 seeds. To run KITTI evaluation without retraining:

```bash
python3 scripts/eval_kitti.py --checkpoint results/run_s42/best_stsie_pid_s42.pt
```

## Training Details

| Parameter | Value |
|-----------|-------|
| Algorithm | PPO with PID-Lagrangian constraint |
| Epochs | 2000 |
| Steps/epoch | 4800 |
| Episode length | 100 steps (dt=0.1s) |
| Action space | 8D (6 Q-scales + 2 R-scales), clipped to [-2, 2] |
| Noise ranges | Q ∈ [0.005, 0.2], R ∈ [0.05, 2.0] (log-uniform) |
| Constraint threshold δ | 0.15 (target: 85% consistency) |
| PID gains | Kp=0.1, Ki=0.008, Kd=0.02 |

## Limitations

- **No real hardware deployment.** KITTI evaluation uses real trajectories with injected synthetic noise. Real sensor noise has correlations and outliers not tested.
- **RMSE tradeoff.** ~2× higher RMSE than overconfident baselines. No downstream planning experiment demonstrating practical benefit.
- **100-step horizon.** Cannot perform minutes-scale adaptation without windowed resets.
- **Position-only observations.** Does not cover IMU/LiDAR multi-sensor fusion.

## License

MIT — see [LICENSE](LICENSE).