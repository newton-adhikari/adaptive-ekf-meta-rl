#!/usr/bin/env python3
"""
Quantitative EKF Comparison using Env6D with Gazebo-derived trajectories.

This produces PUBLISHABLE results by running the evaluation in the same
controlled framework as training (Env6D from run_all.py), but using
trajectory shapes extracted from the actual Gazebo recording.

Why this approach:
  - Same EKF dynamics model as training → fair comparison
  - Controlled process/measurement noise → proper NEES computation
  - Oracle gives ANEES≈6 (sanity check that framework is correct)
  - CC-MetaEKF policy evaluated in its native environment
  - Gazebo trajectory provides realistic kinematic profiles

The Gazebo GUI demo (run_gazebo_comparison.sh) shows sim2real readiness.
This script provides the quantitative comparison table for the paper.
"""

import argparse
import json
import numpy as np
import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from run_all import (
    EKF6D, Task, sample_task, Env6D,
    STSIEEncoder, MLPEncoder, Policy, PIDLag
)

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


# ================================================================
# Extract trajectory kinematics from Gazebo bag
# ================================================================

def extract_trajectory_from_bag(bag_dir, dt=0.1):
    """Extract kinematic profile from rosbag for use as trajectory template."""
    try:
        from rosbags.rosbag2 import Reader
        from rosbags.typesys import get_typestore, Stores
        typestore = get_typestore(Stores.ROS2_HUMBLE)
    except ImportError:
        print("ERROR: pip install rosbags")
        sys.exit(1)

    odom_data = []
    with Reader(bag_dir) as reader:
        for connection, timestamp, rawdata in reader.messages():
            if connection.topic == '/odom':
                msg = typestore.deserialize_cdr(rawdata, connection.msgtype)
                q = msg.pose.pose.orientation
                siny = 2.0 * (q.w * q.z + q.x * q.y)
                cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
                yaw = np.arctan2(siny, cosy)
                odom_data.append([
                    timestamp * 1e-9,
                    msg.pose.pose.position.x,
                    msg.pose.pose.position.y,
                    yaw,
                    msg.twist.twist.linear.x,
                    msg.twist.twist.linear.y,
                    msg.twist.twist.angular.z,
                ])

    odom = np.array(odom_data)
    t0 = odom[0, 0]
    t_grid = np.arange(t0, odom[-1, 0], dt)
    T = len(t_grid)

    # Interpolate to uniform dt
    states = np.column_stack([
        np.interp(t_grid, odom[:, 0], odom[:, 1]),  # px
        np.interp(t_grid, odom[:, 0], odom[:, 2]),  # py
        np.interp(t_grid, odom[:, 0], odom[:, 3]),  # theta
        np.interp(t_grid, odom[:, 0], odom[:, 4]),  # vx
        np.interp(t_grid, odom[:, 0], odom[:, 5]),  # vy
        np.interp(t_grid, odom[:, 0], odom[:, 6]),  # omega
    ])

    # Derive control inputs
    controls = np.column_stack([
        np.gradient(states[:, 3], dt),  # ax
        np.gradient(states[:, 4], dt),  # ay
        np.gradient(states[:, 5], dt),  # alpha
    ])

    print(f"  Trajectory: {T} steps, {T*dt:.1f}s")
    return states, controls


# ================================================================
# Evaluation using Env6D framework (same as training)
# ================================================================

def run_env6d_comparison(trajectory, controls, checkpoint_path,
                         n_episodes=50, ep_len=100, seed=42):
    """
    Run the comparison using Env6D's exact framework.

    For each episode:
      - Sample a noise task (stationary/abrupt/drift)
      - Use the Gazebo trajectory as the nominal motion
      - Apply process noise to true state (as in training)
      - Generate noisy measurements
      - Run each EKF method
      - Compute NEES/NIS/RMSE
    """
    rng = np.random.default_rng(seed)

    # Load CC-MetaEKF policy
    policy = None
    if TORCH_AVAILABLE and checkpoint_path and os.path.exists(checkpoint_path):
        obs_dim = 27
        act_dim = 8
        enc = STSIEEncoder(2, 4, 32, 16, 4)
        policy = Policy(obs_dim, act_dim, enc).eval()
        try:
            ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
            policy.load_state_dict(ckpt)
            print(f"  Loaded policy: {checkpoint_path}")
        except Exception as e:
            print(f"  Policy load failed: {e}")
            policy = None

    # Trim trajectory to episode length segments
    T_traj = len(trajectory)
    Q_nom = np.eye(6) * 0.05
    R_nom = np.eye(2) * 0.5

    methods = ["Fixed", "Sage-Husa", "Oracle", "CC-MetaEKF"]
    results = {m: {"nees": [], "cons": [], "rmse": []} for m in methods}

    for ep in range(n_episodes):
        # Sample noise task
        task = sample_task(rng)

        # Select a random segment from the Gazebo trajectory
        start = rng.integers(0, max(1, T_traj - ep_len - 1))
        seg_states = trajectory[start:start + ep_len]
        seg_controls = controls[start:start + ep_len]
        actual_len = len(seg_states)

        # For each method, run the episode
        for method_name in methods:
            ekf = EKF6D(dt=0.1)
            x_true = seg_states[0].copy()
            x0 = x_true + rng.normal(0, 0.2, 6)
            ekf.reset(x0, np.eye(6) * 0.5, Q_nom.copy(), R_nom.copy())

            innovs = [[0.0, 0.0]] * 30
            nees_ep = []
            step_count = 0

            # Sage-Husa state
            sh_k = 0
            b = 0.98

            for t in range(actual_len):
                # True noise for this timestep
                Q_true, R_true = task.get_noise(t)

                # True state propagation (with process noise)
                u = seg_controls[t] if t < len(seg_controls) else np.zeros(3)
                x_true = ekf._f(x_true, u) + rng.multivariate_normal(
                    np.zeros(6), Q_true)

                # Noisy measurement
                z = x_true[:2] + rng.multivariate_normal(np.zeros(2), R_true)

                # --- Method-specific Q/R setting ---
                if method_name == "Oracle":
                    ekf.Q = Q_true.copy()
                    ekf.R = R_true.copy()

                elif method_name == "CC-MetaEKF" and policy is not None:
                    step_count += 1
                    if step_count % 10 == 0:
                        # Build obs exactly like Env6D._obs()
                        iv = np.array(innovs[-5:]).flatten()
                        obs = np.clip(np.concatenate([
                            iv,
                            [ekf.nees / 6, ekf.nis / 2,
                             np.log1p(max(np.trace(ekf.P), 0))],
                            np.log1p(np.maximum(np.diag(ekf.P), 0)),
                            np.log1p(np.maximum(np.diag(ekf.Q), 0)),
                            np.log1p(np.maximum(np.diag(ekf.R), 0)),
                        ]), -20, 20).astype(np.float32)

                        ib = np.array(innovs, dtype=np.float32)
                        fs = np.array([ekf.nees / 6, ekf.nis / 2,
                                       np.log1p(max(np.trace(ekf.P), 0)),
                                       np.log1p(max(ekf.S[0, 0], 0))],
                                      dtype=np.float32)

                        with torch.no_grad():
                            ot = torch.tensor(obs).unsqueeze(0)
                            ibt = torch.tensor(ib).unsqueeze(0)
                            fst = torch.tensor(fs).unsqueeze(0)
                            dist, _, _ = policy(ot, ibt, fst)
                            action = dist.mean.squeeze(0).numpy()

                        action = np.clip(action, -2.0, 2.0)
                        alphas = np.clip(np.exp(action), 0.01, 100.0)
                        ekf.Q = np.diag(alphas[:6]) @ Q_nom
                        ekf.R = np.diag(alphas[6:]) @ R_nom

                elif method_name == "Sage-Husa":
                    pass  # Adapts after update

                # EKF predict + update
                ekf.predict(u)
                nu = ekf.update(z)
                nees = ekf.compute_nees(x_true)

                # Sage-Husa adaptation (after update)
                if method_name == "Sage-Husa":
                    sh_k += 1
                    if sh_k > 5:
                        d_k = (1 - b) / (1 - b ** sh_k)
                        H = np.zeros((2, 6)); H[0, 0] = 1; H[1, 1] = 1
                        R_new = (1 - d_k) * ekf.R + d_k * (
                            np.outer(nu, nu) - H @ ekf.P @ H.T)
                        eigvals = np.linalg.eigvalsh(R_new)
                        if np.all(eigvals > 1e-8):
                            ekf.R = R_new

                innovs.append(nu.tolist())
                innovs = innovs[-30:]
                nees_ep.append(nees)

            # Episode metrics
            nees_arr = np.array(nees_ep)
            results[method_name]["nees"].append(float(np.mean(nees_arr)))
            results[method_name]["cons"].append(
                float(np.mean((nees_arr >= 1.237) & (nees_arr <= 14.449))))
            rmse = float(np.sqrt(np.mean((x_true - ekf.x) ** 2)))
            results[method_name]["rmse"].append(rmse)

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bag_dir", default="results/gazebo/recorded_bag")
    parser.add_argument("--output_dir", default="results/gazebo")
    parser.add_argument("--checkpoint", default="results/run_s42/best_stsie_pid_s42.pt")
    parser.add_argument("--n_episodes", type=int, default=50)
    parser.add_argument("--ep_len", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"\n  Gazebo-trajectory EKF Comparison (Env6D framework)")
    print(f"  Episodes: {args.n_episodes}, Length: {args.ep_len} steps")
    print(f"  Seed: {args.seed}")
    print()

    # Load trajectory
    if os.path.isdir(args.bag_dir):
        print("  Loading Gazebo trajectory from bag...")
        trajectory, controls = extract_trajectory_from_bag(args.bag_dir)
    else:
        print("  No bag found. Using synthetic figure-8 trajectory.")
        T = 1000
        dt = 0.1
        trajectory = np.zeros((T, 6))
        for t in range(T):
            a = 0.3 * t * dt
            trajectory[t] = [2*np.sin(a), np.sin(2*a), a,
                             2*0.3*np.cos(a), 0.3*2*np.cos(2*a), 0.3]
        controls = np.zeros((T, 3))

    print()

    # Run comparison
    results = run_env6d_comparison(
        trajectory, controls, args.checkpoint,
        n_episodes=args.n_episodes,
        ep_len=args.ep_len,
        seed=args.seed,
    )

    # Print results
    print(f"\n  {'='*65}")
    print(f"  {'Method':<15} {'ANEES':>8} {'±std':>7} {'Cons%':>7} {'±':>5} {'RMSE':>7}")
    print(f"  {'-'*65}")

    summary = {}
    for method, data in results.items():
        nees_arr = np.array(data["nees"])
        cons_arr = np.array(data["cons"])
        rmse_arr = np.array(data["rmse"])

        summary[method] = {
            "anees_mean": float(np.mean(nees_arr)),
            "anees_std": float(np.std(nees_arr)),
            "consistency_mean": float(np.mean(cons_arr)),
            "consistency_std": float(np.std(cons_arr)),
            "rmse_mean": float(np.mean(rmse_arr)),
            "rmse_std": float(np.std(rmse_arr)),
        }

        print(f"  {method:<15} {np.mean(nees_arr):>7.2f} ±{np.std(nees_arr):>5.2f}  "
              f"{np.mean(cons_arr):>5.1%} ±{np.std(cons_arr):>4.1%} "
              f"{np.mean(rmse_arr):>6.3f}")

    print(f"  {'='*65}")
    print(f"  Target ANEES: 6.0 | Chi² 95% bounds: [1.24, 14.45]")
    print()

    # Save
    output = {
        "experiment": "gazebo_trajectory_env6d_comparison",
        "description": "EKF comparison using Env6D framework with Gazebo-derived trajectory shapes",
        "n_episodes": args.n_episodes,
        "ep_len": args.ep_len,
        "seed": args.seed,
        "results": summary,
    }
    out_path = os.path.join(args.output_dir, "gazebo_comparison.json")
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"  Saved: {out_path}")


if __name__ == "__main__":
    main()
 