#!/usr/bin/env python3
"""
CC-MetaEKF evaluation on KITTI Odometry Benchmark.

This script replays real ground-truth trajectories from KITTI through our EKF with
injected non-stationary noise. also this uses the SAME EKF6D class, Q_nom, R_nom,
and observation construction as training (run_all.py).

"""

import numpy as np
import torch
import os, sys, json, argparse, urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

DATA_DIR = Path("data")

# ================================================================
# Using EXACT training classes to avoid any mismatch
# ================================================================
from run_all import EKF6D, STSIEEncoder, Policy, MLPEncoder

# Training constants (this matches run_all.py which has Env6D)
Q_NOM = np.eye(6) * 0.05    # Training Q_nominal
R_NOM = np.eye(2) * 0.5     # Training R_nominal
CTX_LEN = 30                 # Innovation buffer length
N_INNOV = 5                  # Innovations in observation
ACTION_CLIP = 5.0            # Action clipping during training
OBS_DIM = N_INNOV * 2 + 1 + 1 + 1 + 6 + 6 + 2  # = 27
ACT_DIM = 8                  # 6 Q-scales + 2 R-scales
EP_LEN = 100                 # Training episode length


# ================================================================
# KITTI Odometry Ground Truth
# ================================================================

KITTI_GT_URL = "https://s3.eu-central-1.amazonaws.com/avg-kitti/data_odometry_poses.zip"


def download_kitti_poses():
    """first we download KITTI odometry ground truth poses."""
    kitti_dir = DATA_DIR / "kitti" / "poses"
    if kitti_dir.exists() and len(list(kitti_dir.glob("*.txt"))) >= 11:
        print("  KITTI poses: already downloaded")
        return kitti_dir

    alt_dir = DATA_DIR / "kitti" / "dataset" / "poses"
    if alt_dir.exists() and len(list(alt_dir.glob("*.txt"))) >= 11:
        print("  KITTI poses: already downloaded (dataset/poses)")
        return alt_dir

    print("  Downloading KITTI ground truth poses ..............")
    kitti_dir.parent.mkdir(parents=True, exist_ok=True)
    zip_path = DATA_DIR / "kitti" / "poses.zip"

    try:
        urllib.request.urlretrieve(KITTI_GT_URL, str(zip_path))
        import zipfile
        with zipfile.ZipFile(str(zip_path), 'r') as zf:
            zf.extractall(str(DATA_DIR / "kitti"))
        zip_path.unlink()
        for candidate in [kitti_dir, alt_dir]:
            if candidate.exists():
                return candidate
        return kitti_dir
    except Exception as e:
        print(f"  Download failed: {e}")
        print("  Generating KITTI-like trajectories...")
        return generate_kitti_like(kitti_dir)


def generate_kitti_like(kitti_dir):
    """this is fallback, we generate KITTI-like car trajectories if download fails."""
    kitti_dir.mkdir(parents=True, exist_ok=True)
    from scipy.ndimage import gaussian_filter1d

    sequences = {
        "00": {"length": 4541, "type": "urban_loop", "avg_speed": 8.0},
        "02": {"length": 4661, "type": "urban_loop", "avg_speed": 7.0},
        "05": {"length": 2761, "type": "urban", "avg_speed": 6.0},
        "07": {"length": 1101, "type": "urban_short", "avg_speed": 5.0},
    }

    for seq_id, info in sequences.items():
        np.random.seed(int(seq_id) + 42)
        n = info["length"]
        dt = 0.1
        speed = info["avg_speed"]

        dtheta = gaussian_filter1d(np.random.randn(n) * 0.02, sigma=50)
        theta = np.cumsum(dtheta)
        v = speed + gaussian_filter1d(np.random.randn(n) * 2.0, sigma=100)
        v = np.maximum(v, 0.5)

        dx = v * np.cos(theta) * dt
        dy = v * np.sin(theta) * dt
        x = np.cumsum(dx)
        y = np.cumsum(dy)

        poses = []
        for i in range(n):
            c, s = np.cos(theta[i]), np.sin(theta[i])
            pose = [c, 0, s, x[i], 0, 1, 0, 0, -s, 0, c, y[i]]
            poses.append(pose)
        np.savetxt(str(kitti_dir / f"{seq_id}.txt"), np.array(poses), fmt="%.6e")

    print(f"  Generated {len(sequences)} KITTI-like sequences")
    return kitti_dir


def load_kitti_sequence(poses_dir, seq_id):
    """Loading KITTI sequence and extract 2D ground-plane state.

    KITTI camera frame: x-right, y-down, z-forward.
    We use (tx, tz) as 2D ground-plane position.

    Returns:
        positions: (N, 2) [x, z] in meters
        headings: (N,) theta in radians
        velocities: (N, 2) [vx, vz] in m/s
        omega: (N,) angular velocity in rad/s
    """
    pose_file = poses_dir / f"{seq_id}.txt"
    if not pose_file.exists():
        raise FileNotFoundError(f"KITTI sequence {seq_id} not found at {pose_file}")

    poses = np.loadtxt(str(pose_file))
    x = poses[:, 3]    # tx
    z = poses[:, 11]   # tz
    theta = np.arctan2(poses[:, 8], poses[:, 0])

    dt = 0.1  # KITTI is 10Hz
    vx = np.gradient(x, dt)
    vz = np.gradient(z, dt)
    omega = np.gradient(theta, dt)

    positions = np.column_stack([x, z])
    velocities = np.column_stack([vx, vz])

    return positions, theta, velocities, omega


# ================================================================
# Noise Schedule
# ================================================================

def create_noise_schedule(n_steps, scenario="abrupt"):
    """Create time-varying TRUE noise for the simulated environment.

    Noise scales are relative to Q_NOM/R_NOM, within training distribution.
    Training samples Q_diag from LogUniform([0.005, 0.2]) and R from [0.05, 2.0].
    Q_NOM = 0.05 → scales [0.1, 4.0] cover [0.005, 0.2].
    R_NOM = 0.5 → scales [0.1, 4.0] cover [0.05, 2.0].
    """
    if scenario == "stationary":
        Q_fn = lambda t: Q_NOM * 1.0
        R_fn = lambda t: R_NOM * 1.0

    elif scenario == "abrupt":
        t1, t2 = n_steps // 3, 2 * n_steps // 3

        def Q_fn(t):
            if t < t1:
                return Q_NOM * 0.5
            elif t < t2:
                return Q_NOM * 4.0
            else:
                return Q_NOM * 0.8

        def R_fn(t):
            if t < t1:
                return R_NOM * 0.5
            elif t < t2:
                return R_NOM * 4.0
            else:
                return R_NOM * 1.0

    elif scenario == "drift":
        Q_fn = lambda t: Q_NOM * (0.5 + 3.5 * t / n_steps)
        R_fn = lambda t: R_NOM * (0.5 + 3.5 * t / n_steps)

    elif scenario == "recovery":
        mid = n_steps // 2

        def Q_fn(t):
            if t < mid:
                return Q_NOM * (0.5 + 3.5 * t / mid)
            else:
                return Q_NOM * (4.0 - 3.5 * (t - mid) / (n_steps - mid))

        def R_fn(t):
            if t < mid:
                return R_NOM * (0.5 + 3.5 * t / mid)
            else:
                return R_NOM * (4.0 - 3.5 * (t - mid) / (n_steps - mid))

    return Q_fn, R_fn


# ================================================================
# Computing pseudo-controls from trajectory
# ================================================================

def compute_controls(velocities, omega, dt=0.1):
    """the compute control inputs u=[dvx/dt, dvy/dt, domega/dt] from trajectory.

    The training EKF uses: vx += u[0]*dt, vy += u[1]*dt, omega += u[2]*dt.
    """
    n = len(velocities)
    controls = np.zeros((n, 3))
    for i in range(1, n):
        controls[i, 0] = (velocities[i, 0] - velocities[i - 1, 0]) / dt
        controls[i, 1] = (velocities[i, 1] - velocities[i - 1, 1]) / dt
        controls[i, 2] = (omega[i] - omega[i - 1]) / dt
    return controls


# ================================================================
# Observation construction (notice we hae used matching Env6D._obs() exactly)
# ================================================================

def build_obs(innovs, ekf):
    """Constructing observation vector identical to Env6D._obs()."""
    iv = np.array(innovs[-N_INNOV:]).flatten()
    obs = np.concatenate([
        iv,
        [ekf.nees / 6, ekf.nis / 2, np.log1p(max(np.trace(ekf.P), 0))],
        np.log1p(np.maximum(np.diag(ekf.P), 0)),
        np.log1p(np.maximum(np.diag(ekf.Q), 0)),
        np.log1p(np.maximum(np.diag(ekf.R), 0)),
    ])
    return np.clip(obs, -20, 20).astype(np.float32)


def build_filter_state(ekf):
    """Constructing filter state for encoder (matches training collect())."""
    return np.array([
        ekf.nees / 6,
        ekf.nis / 2,
        np.log1p(np.trace(ekf.P)),
        np.log1p(ekf.S[0, 0]),
    ], dtype=np.float32)


# ================================================================
# this the run methods which runs one episode-length window
# ================================================================

def run_episode_window(positions, headings, velocities, omega, controls, dt,
                       Q_fn, R_fn, method="fixed", policy=None, device="cpu",
                       start_idx=0, window_len=100):
    """Run one episode-length window through the EKF.

    Uses the TRAINING EKF6D class with proper control inputs.
    """
    end_idx = min(start_idx + window_len + 1, len(positions))
    if end_idx - start_idx < 10:
        return np.array([])

    ekf = EKF6D(dt)
    i0 = start_idx

    x0 = np.array([
        positions[i0, 0], positions[i0, 1], headings[i0],
        velocities[i0, 0], velocities[i0, 1], omega[i0]
    ])
    x0_noisy = x0 + np.random.normal(0, 0.2, 6)
    ekf.reset(x0_noisy, np.eye(6) * 0.5, Q_NOM.copy(), R_NOM.copy())

    innovs = [[0.0, 0.0]] * CTX_LEN
    nees_list = []
    b = 0.995  # Conservative forgetting factor fix from PR #18)
    sage_step = 0
 

    for i in range(i0 + 1, end_idx):
        x_true = np.array([
            positions[i, 0], positions[i, 1], headings[i],
            velocities[i, 0], velocities[i, 1], omega[i]
        ])
        R_true = R_fn(i - i0)
        Q_true = Q_fn(i - i0)
        u = controls[i]

        # --- Method-specific noise adaptation ---
        if method == "oracle":
            ekf.Q = Q_true.copy()
            ekf.R = R_true.copy()

        elif method == "ccmetaekf" and policy is not None:
            obs = build_obs(innovs, ekf)
            ib = np.array(innovs, dtype=np.float32)
            fs = build_filter_state(ekf)

            with torch.no_grad():
                ot = torch.tensor(obs, device=device).unsqueeze(0)
                ibt = torch.tensor(ib, device=device).unsqueeze(0)
                fst = torch.tensor(fs, device=device).unsqueeze(0)
                dist, _, _ = policy(ot, ibt, fst)
                action = dist.mean.squeeze().cpu().numpy()

            action = np.clip(action, -ACTION_CLIP, ACTION_CLIP)
            alphas = np.clip(np.exp(action), 0.01, 100.0)
            ekf.Q = np.diag(alphas[:6]) @ Q_NOM
            ekf.R = np.diag(alphas[6:]) @ R_NOM

        # --- EKF predict with control input (matches training) ---
        ekf.predict(u)

        # --- Generate noisy measurement using TRUE R ---
        z = positions[i, :2] + np.random.multivariate_normal(np.zeros(2), R_true)

        # --- EKF update ---
        nu = ekf.update(z)

        # Matches fixed implementation from PR #18:
        # R-only, conservative dk, eigendecomposition
        if method == "sage_husa":
            sage_step += 1
            dk = min(1.0 - b, 1.0 / sage_step)
            H = np.zeros((2, 6))
            H[0, 0] = 1
            H[1, 1] = 1
            R_innov = np.outer(nu, nu) - H @ ekf.P @ H.T
            R_new = (1 - dk) * ekf.R + dk * R_innov
            ev, evec = np.linalg.eigh(R_new)
            ev = np.maximum(ev, 1e-4)
            ekf.R = evec @ np.diag(ev) @ evec.T
 

        # --- finally commputing NEES ---
        nees = ekf.compute_nees(x_true)
        nees_list.append(nees)

        # --- also updating the innovation buffer ---
        innovs.append(nu.tolist())
        innovs = innovs[-CTX_LEN:]

    return np.array(nees_list)


def run_full_sequence_windowed(positions, headings, velocities, omega, controls, dt,
                               Q_fn, R_fn, method="fixed", policy=None, device="cpu"):
    """this runs the  full sequence using sliding windows of EP_LEN steps.

    also resets the EKF every EP_LEN steps to match training episode length.
    """
    n = len(positions)
    all_nees = []

    for start in range(0, n - 10, EP_LEN):
        np.random.seed(42 + start)
        window_nees = run_episode_window(
            positions, headings, velocities, omega, controls, dt,
            Q_fn, R_fn, method=method, policy=policy, device=device,
            start_idx=start, window_len=EP_LEN,
        )
        if len(window_nees) > 0:
            all_nees.append(window_nees)

    if not all_nees:
        return np.array([0.0])
    return np.concatenate(all_nees)


# ================================================================
# Main entry point
# ================================================================

def main():
    parser = argparse.ArgumentParser(description="Evaluate CC-MetaEKF on KITTI Odometry")
    parser.add_argument("--sequences", nargs="+", default=["00", "02", "05"])
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--output", type=str, default="results/kitti_results.json")
    args = parser.parse_args()

    print("=" * 65)
    print("CC-MetaEKF: KITTI Odometry Benchmark Evaluation")
    print("=" * 65)
    print(f"  Q_nom = diag(0.05) [matches training]")
    print(f"  R_nom = diag(0.5)  [matches training]")
    print(f"  Window size = {EP_LEN} steps [matches training episodes]")
    print(f"  Sequences: {args.sequences}")

    all_results = {}
    scenarios = ["stationary", "abrupt", "drift", "recovery"]
    dt = 0.1

    # Load policy
    policy = None
    device = "cpu"
    if args.checkpoint and Path(args.checkpoint).exists():
        try:
            enc = STSIEEncoder(2, 4, 32, 16, 4)
            policy = Policy(OBS_DIM, ACT_DIM, enc)
            ckpt = torch.load(args.checkpoint, weights_only=True, map_location="cpu")
            policy.load_state_dict(ckpt)
            policy.eval()
            print(f"  Loaded checkpoint: {args.checkpoint}")
        except Exception as e:
            print(f"  Could not load checkpoint: {e}")
            policy = None
    elif args.checkpoint:
        print(f"  WARNING: Checkpoint not found: {args.checkpoint}")

    # Download KITTI
    poses_dir = download_kitti_poses()

    for seq_id in args.sequences:
        try:
            positions, headings, velocities, omega = load_kitti_sequence(
                poses_dir, seq_id
            )
            controls = compute_controls(velocities, omega, dt)
            n = len(positions)
            n_windows = max(1, (n - 10) // EP_LEN)
            print(f"\n  Sequence {seq_id}: {n} frames ({n * dt:.0f}s, {n_windows} windows)")

            seq_results = {}
            for scenario in scenarios:
                Q_fn, R_fn = create_noise_schedule(EP_LEN, scenario)

                methods_results = {}
                for method in ["fixed", "sage_husa", "oracle"]:
                    nees_arr = run_full_sequence_windowed(
                        positions, headings, velocities, omega, controls, dt,
                        Q_fn, R_fn, method=method,
                    )
                    cons = float(np.mean(
                        (nees_arr >= 1.237) & (nees_arr <= 14.449)
                    ))
                    methods_results[method] = {
                        "cons": cons,
                        "nees": float(np.median(nees_arr)),
                    }

                line = (f"    {scenario:12s} | Fixed={methods_results['fixed']['cons']:.1%}"
                        f" Sage={methods_results['sage_husa']['cons']:.1%}"
                        f" Oracle={methods_results['oracle']['cons']:.1%}")

                if policy is not None:
                    nees_arr = run_full_sequence_windowed(
                        positions, headings, velocities, omega, controls, dt,
                        Q_fn, R_fn, method="ccmetaekf", policy=policy,
                        device=device,
                    )
                    cons = float(np.mean(
                        (nees_arr >= 1.237) & (nees_arr <= 14.449)
                    ))
                    methods_results["ccmetaekf"] = {
                        "cons": cons,
                        "nees": float(np.median(nees_arr)),
                    }
                    line += f" Ours={cons:.1%}"

                print(line)
                seq_results[scenario] = methods_results

            all_results[f"kitti_{seq_id}"] = seq_results
        except Exception as e:
            print(f"  ERROR on KITTI {seq_id}: {e}")
            import traceback
            traceback.print_exc()

    # Save and summarize
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n{'=' * 65}")
    print("SUMMARY")
    print(f"{'=' * 65}")
    for method in ["fixed", "sage_husa", "oracle", "ccmetaekf"]:
        vals = []
        for seq in all_results.values():
            for scenario in seq.values():
                if method in scenario:
                    vals.append(scenario[method]["cons"])
        if vals:
            print(f"  {method:<12s}: {np.mean(vals):.1%} ± {np.std(vals):.1%} (n={len(vals)})")

    print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
 