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
# Training
# ================================================================

def train_kalmannet(n_episodes=5000, lr=1e-3, seed=42,
                    save_path="checkpoints/kalmannet.pt", tbptt_len=20):
    """Train KalmanNet with supervised learning using oracle Kalman gains.

    Generates training data by running the EKF environment, computing the
    oracle Kalman gain (using true Q/R) at each step, and training the GRU
    to predict these gains from observable quantities (innovation, P, S).

    Uses truncated BPTT (tbptt_len steps) for memory efficiency.
    """
    from run_all import Env6D, EKF6D, sample_task

    np.random.seed(seed)
    torch.manual_seed(seed)

    print("=" * 60)
    print("KalmanNet Training (supervised with oracle K)")
    print("=" * 60)
    print(f"  Episodes: {n_episodes}, LR: {lr}, TBPTT: {tbptt_len}")

    state_dim, meas_dim = 6, 2
    model = KalmanNetGRU(state_dim, meas_dim, hidden_dim=64, n_layers=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    env = Env6D(ep_len=100)
    rng = np.random.default_rng(seed)

    losses = []
    best_loss = float('inf')

    for ep in range(n_episodes):
        task = sample_task(rng)
        model.reset()
        model.train()

        env.rng = np.random.default_rng(seed + ep)
        env.reset(task=task)

        # Collect oracle gains and inputs for the full episode
        ep_innovations = []
        ep_P_diags = []
        ep_S_diags = []
        ep_K_targets = []

        for t in range(env.ep_len):
            Q_true, R_true = task.get_noise(t)
            H = np.zeros((meas_dim, state_dim))
            H[0, 0] = 1
            H[1, 1] = 1

            # Oracle gain: K = P_pred @ H^T @ inv(H @ P_pred @ H^T + R_true)
            # P_pred is the covariance AFTER predict, BEFORE update
            P_pred = env.ekf.P.copy()
            S_oracle = H @ P_pred @ H.T + R_true
            K_oracle = P_pred @ H.T @ np.linalg.inv(S_oracle)

            # Step environment (this does predict + update internally)
            action = np.zeros(8)
            obs, ctx, _, done, info = env.step(action)

            # After step, innovation is available
            innovation = np.array(env.innovs[-1], dtype=np.float32)
            P_diag = np.log1p(np.maximum(np.diag(env.ekf.P), 0)).astype(np.float32)
            S_diag = np.log1p(np.maximum(np.diag(env.ekf.S), 0)).astype(np.float32)

            ep_innovations.append(innovation)
            ep_P_diags.append(P_diag)
            ep_S_diags.append(S_diag)
            ep_K_targets.append(K_oracle.astype(np.float32))

            if done:
                break

        # Train with truncated BPTT
        T = len(ep_innovations)
        ep_loss = 0.0
        n_chunks = 0

        for start in range(0, T, tbptt_len):
            end = min(start + tbptt_len, T)
            chunk_loss = torch.tensor(0.0)

            for t in range(start, end):
                inn_t = torch.tensor(ep_innovations[t]).unsqueeze(0)
                P_t = torch.tensor(ep_P_diags[t]).unsqueeze(0)
                S_t = torch.tensor(ep_S_diags[t]).unsqueeze(0)

                K_pred = model(inn_t, P_t, S_t)
                K_target = torch.tensor(ep_K_targets[t]).unsqueeze(0)

                chunk_loss = chunk_loss + loss_fn(K_pred, K_target)

            # Backprop through this chunk
            optimizer.zero_grad()
            (chunk_loss / (end - start)).backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            # Detach hidden state for next chunk (truncated BPTT)
            model.hidden = model.hidden.detach() if model.hidden is not None else None

            ep_loss += chunk_loss.item()
            n_chunks += 1

        if n_chunks > 0:
            losses.append(ep_loss / T)

        # Logging
        if (ep + 1) % 500 == 0:
            avg_loss = np.mean(losses[-500:])
            print(f"  Ep {ep+1:>5d}/{n_episodes} | Loss: {avg_loss:.6f}")
            if avg_loss < best_loss:
                best_loss = avg_loss
                Path(save_path).parent.mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), save_path)

    # Final save
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"\n  Final loss: {np.mean(losses[-100:]):.6f}")
    print(f"  Saved to: {save_path}")
    return model


# ================================================================
# Evaluation
# ================================================================

def evaluate_kalmannet(model, positions, headings, velocities, omega, controls,
                       dt, Q_nom, R_nom, Q_fn, R_fn, window_len=100, start_idx=0):
    """Evaluate KalmanNet on a trajectory window.

    KalmanNet predicts K directly. We maintain P for NEES computation but
    note that this P is not derived from K — it's maintained separately
    using Q_nom. This means the NEES may not be statistically valid for
    KalmanNet (this is the core limitation vs CC-MetaEKF).
    """
    state_dim, meas_dim = 6, 2
    end_idx = min(start_idx + window_len + 1, len(positions))
    if end_idx - start_idx < 10:
        return np.array([])

    i0 = start_idx

    # Initialize
    x = np.array([positions[i0, 0], positions[i0, 1], headings[i0],
                  velocities[i0, 0], velocities[i0, 1], omega[i0]])
    x = x + np.random.normal(0, 0.2, 6)
    P = np.eye(6) * 0.5

    model.eval()
    model.reset()
    nees_list = []

    for i in range(i0 + 1, end_idx):
        x_true = np.array([positions[i, 0], positions[i, 1], headings[i],
                           velocities[i, 0], velocities[i, 1], omega[i]])
        R_true = R_fn(i - i0)
        u = controls[i]

        # --- Predict (same motion model as training EKF6D) ---
        th, vx, vy = x[2], x[3], x[4]
        F = np.eye(6)
        F[0, 2] = (-vx*np.sin(th) - vy*np.cos(th))*dt
        F[0, 3] = np.cos(th)*dt; F[0, 4] = -np.sin(th)*dt
        F[1, 2] = (vx*np.cos(th) - vy*np.sin(th))*dt
        F[1, 3] = np.sin(th)*dt; F[1, 4] = np.cos(th)*dt
        F[2, 5] = dt

        # State prediction with control input
        x[0] += (vx*np.cos(th) - vy*np.sin(th)) * dt
        x[1] += (vx*np.sin(th) + vy*np.cos(th)) * dt
        x[2] += x[5] * dt
        x[3] += u[0] * dt
        x[4] += u[1] * dt
        x[5] += u[2] * dt

        P = F @ P @ F.T + Q_nom
        P = (P + P.T) / 2

        # --- Measurement ---
        H = np.zeros((2, 6)); H[0, 0] = 1; H[1, 1] = 1
        z = positions[i, :2] + np.random.multivariate_normal(np.zeros(2), R_true)
        nu = z - H @ x
        S = H @ P @ H.T + R_nom

        # --- KalmanNet: predict gain ---
        with torch.no_grad():
            inn_t = torch.tensor(nu, dtype=torch.float32).unsqueeze(0)
            P_diag_t = torch.tensor(
                np.log1p(np.maximum(np.diag(P), 0)), dtype=torch.float32
            ).unsqueeze(0)
            S_diag_t = torch.tensor(
                np.log1p(np.maximum(np.diag(S), 0)), dtype=torch.float32
            ).unsqueeze(0)
            K = model(inn_t, P_diag_t, S_diag_t).squeeze(0).numpy()

        # --- Update with learned gain ---
        x = x + K @ nu
        I_KH = np.eye(6) - K @ H
        P = I_KH @ P @ I_KH.T + K @ R_nom @ K.T
        P = (P + P.T) / 2

        # --- NEES ---
        e = x_true - x
        try:
            nees = float(e @ np.linalg.inv(P) @ e)
        except:
            nees = 100.0
        nees_list.append(nees)

    return np.array(nees_list)


def evaluate_on_kitti(checkpoint_path, sequences=["00", "02", "05", "07"]):
    """Run KalmanNet evaluation on KITTI """
    sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
    from eval_kitti import (
        download_kitti_poses, load_kitti_sequence, compute_controls,
        create_noise_schedule, Q_NOM, R_NOM, EP_LEN
    )

    print("=" * 60)
    print("KalmanNet: KITTI Evaluation")
    print("=" * 60)

    # Load model
    model = KalmanNetGRU(6, 2, hidden_dim=64, n_layers=2)
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu", weights_only=True))
    model.eval()
    print(f"  Loaded: {checkpoint_path}")

    poses_dir = download_kitti_poses()
    scenarios = ["stationary", "abrupt", "drift", "recovery"]
    dt = 0.1

    all_cons = []

    for seq_id in sequences:
        positions, headings, velocities, omega = load_kitti_sequence(poses_dir, seq_id)
        controls = compute_controls(velocities, omega, dt)
        n = len(positions)
        print(f"\n  Sequence {seq_id}: {n} frames ({n*dt:.0f}s)")

        for scenario in scenarios:
            Q_fn, R_fn = create_noise_schedule(EP_LEN, scenario)

            # Windowed evaluation (same as eval_kitti.py)
            all_nees = []
            for start in range(0, n - 10, EP_LEN):
                np.random.seed(42 + start)
                model.reset()
                window_nees = evaluate_kalmannet(
                    model, positions, headings, velocities, omega, controls,
                    dt, Q_NOM, R_NOM, Q_fn, R_fn,
                    window_len=EP_LEN, start_idx=start,
                )
                if len(window_nees) > 0:
                    all_nees.append(window_nees)

            if all_nees:
                nees_arr = np.concatenate(all_nees)
                cons = float(np.mean((nees_arr >= 1.237) & (nees_arr <= 14.449)))
                all_cons.append(cons)
                print(f"    {scenario:12s} | KalmanNet={cons:.1%}")

    if all_cons:
        print(f"\n  MEAN: {np.mean(all_cons):.1%} ± {np.std(all_cons):.1%}")


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

    if args.train:
        train_kalmannet(n_episodes=args.episodes, save_path=args.checkpoint)
    elif args.eval_kitti:
        evaluate_on_kitti(args.checkpoint)
    else:
        print("Usage:")
        print("  Train:  python3 baselines/kalmannet.py --train --episodes 5000")
        print("  Eval:   python3 baselines/kalmannet.py --eval-kitti --checkpoint checkpoints/kalmannet.pt")
 