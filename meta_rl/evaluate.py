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

    config = load_config(args.config)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Environment
    env = LightweightEKFEnv()

    # Load agent
    agent_config = {"device": "cpu", "encoder_type": "st_sie"}
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    agent = PIDCCPOAgent(
        state_dim=obs_dim, action_dim=act_dim,
        config=agent_config,
    )
    if Path(args.checkpoint).exists():
        agent.load(args.checkpoint)
        print(f"Loaded checkpoint: {args.checkpoint}")

    # Setup baselines
    Q_nom = np.eye(6) * 0.1
    R_nom = np.eye(2) * 1.0
    baselines = {
        "Fixed EKF": FixedEKFAdapter(Q_nom, R_nom),
        "Sage-Husa": SageHusaAdapter(Q_nom, R_nom),
        "Innovation-Based": InnovationAdaptiveAdapter(Q_nom, R_nom),
        "VB-EKF": VBEKFAdapter(Q_nom, R_nom),
        "RLS Covariance": RLSCovarianceAdapter(Q_nom, R_nom),
        "Oracle EKF": OracleEKFAdapter(),
    }

    # Test tasks
    sampler = TaskSampler()
    rng = np.random.default_rng(123)
    test_tasks = sampler.sample_batch(
        config.get("evaluation", {}).get("n_eval_episodes", 50), rng
    )

    all_results = {}

    # Evaluate baselines
    for name, adapter in baselines.items():
        print(f"Evaluating: {name}")
        task_results = []
        for task in test_tasks:
            adapter.reset()
            if name == "Oracle EKF":
                adapter.set_noise_schedule(task.get_noise)
            result = run_baseline_episode(env, adapter, task)
            task_results.append(result)

        avg_consistency = np.mean([r["consistency_rate"] for r in task_results])
        avg_anees = np.mean([r["anees"] for r in task_results])
        all_results[name] = {
            "consistency_rate": float(avg_consistency),
            "anees": float(avg_anees),
        }
        print(f"  Consistency: {avg_consistency:.3f} | ANEES: {avg_anees:.3f}")

    # Evaluate CC-MetaEKF
    print("Evaluating: CC-MetaEKF")
    task_results = []
    for task in test_tasks:
        result = run_agent_episode(env, agent, task)
        task_results.append(result)

    avg_consistency = np.mean([r["consistency_rate"] for r in task_results])
    avg_anees = np.mean([r["anees"] for r in task_results])
    all_results["CC-MetaEKF"] = {
        "consistency_rate": float(avg_consistency),
        "anees": float(avg_anees),
    }
    print(f"  Consistency: {avg_consistency:.3f} | ANEES: {avg_anees:.3f}")

    # Save results
    with open(output_dir / "evaluation_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\nResults saved to {output_dir / 'evaluation_results.json'}")


if __name__ == "__main__":
    main()
