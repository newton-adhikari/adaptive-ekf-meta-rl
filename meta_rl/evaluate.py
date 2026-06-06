"""
CC-MetaEKF evaluation script.
This runs all baselines and ablations for different test scenarios.

python -m meta_rl.evaluate --checkpoint checkpoints/best_model.pt --config configs/eval_config.yaml
"""

import argparse
import json
import yaml
import numpy as np
import torch
from pathlib import Path

from meta_rl.utils.ekf import EKF
from meta_rl.utils.metrics import (
    compute_nees, compute_nis, consistency_rate,
    compute_rmse, compute_position_rmse, chi2_bounds,
)
from meta_rl.envs.lightweight_ekf_env import LightweightEKFEnv
from meta_rl.envs.task_sampler import TaskSampler, TaskConfig
from meta_rl.agents.pid_ccpo_agent import PIDCCPOAgent

from baselines.fixed_ekf import FixedEKFAdapter
from baselines.sage_husa import SageHusaAdapter
from baselines.innovation_adaptive import InnovationAdaptiveAdapter
from baselines.variational_bayes_ekf import VBEKFAdapter
from baselines.rls_covariance import RLSCovarianceAdapter
from baselines.oracle_ekf import OracleEKFAdapter



def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)
    

def run_baseline_episode(
    env: LightweightEKFEnv,
    adapter,
    task: TaskConfig,
) -> dict:
    """Run one episode with a baseline adapter."""
    obs, info = env.reset()
    env._task = task

    nees_values = []
    nis_values = []
    done = False
    step = 0

    while not done:
        # Get EKF state for adapter
        innovation = env.ekf.state.innovation
        P = env.ekf.state.P
        S = env.ekf.state.S
        H = env.ekf._measurement_jacobian(env.ekf.state.x)

        # Adapt Q/R
        Q_new, R_new = adapter.adapt(
            innovation=innovation, P=P, S=S, H=H,
        )
        env.ekf.set_noise(Q_new, R_new)

        # Step with identity action (adapter handles Q/R)
        action = np.ones(env.action_space.shape[0])
        obs, reward, done, truncated, info = env.step(action)

        nees_values.append(info.get("nees", 0.0))
        nis_values.append(info.get("nis", 0.0))
        step += 1
        if truncated:
            break

    nees_arr = np.array(nees_values)
    return {
        "nees_values": nees_arr.tolist(),
        "anees": float(np.mean(nees_arr)),
        "consistency_rate": consistency_rate(nees_arr, env.state_dim),
    }


def run_agent_episode(
    env: LightweightEKFEnv,
    agent: PIDCCPOAgent,
    task: TaskConfig,
) -> dict:
    """Run one episode with the trained CC-MetaEKF agent."""
    obs, info = env.reset()
    env._task = task

    nees_values = []
    done = False

    while not done:
        innov_window = env.ekf.get_innovation_window(env.innovation_window)
        filter_state = np.array([
            env.ekf.state.nees, env.ekf.state.nis,
            np.trace(env.ekf.state.P), *np.diag(env.ekf.state.S),
        ])
        action = agent.select_action(
            obs, innov_window, filter_state, deterministic=True
        )
        obs, reward, done, truncated, info = env.step(action)
        nees_values.append(info.get("nees", 0.0))
        if truncated:
            break

    nees_arr = np.array(nees_values)
    return {
        "nees_values": nees_arr.tolist(),
        "anees": float(np.mean(nees_arr)),
        "consistency_rate": consistency_rate(nees_arr, env.state_dim),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pt")
    parser.add_argument("--config", type=str, default="configs/eval_config.yaml")
    parser.add_argument("--output_dir", type=str, default="results/")
    args = parser.parse_args()

if __name__ == "__main__":
    main()
