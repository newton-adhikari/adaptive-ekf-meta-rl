"""
CC-MetaEKF training script.


python -m meta_rl.train --config configs/train_config.yaml --seed 42
"""

import argparse
import os
import time
import yaml
import numpy as np

from meta_rl.agents.replay_buffer import MultiTaskReplayBuffer, Transition


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def collect_episode(env, agent, task_id, deterministic=False):
    obs, info = env.reset()
    transitions = []
    done = False
    while not done:
        iw = env.ekf.get_innovation_window(env.innovation_window)
        fs = np.array([
            env.ekf.state.nees, env.ekf.state.nis,
            np.trace(env.ekf.state.P), *np.diag(env.ekf.state.S),
        ])
        action = agent.select_action(obs, iw, fs, deterministic=deterministic)
        next_obs, reward, done, truncated, info = env.step(action)
        transitions.append(Transition(
            state=obs, action=action, reward=reward,
            next_state=next_obs, done=done or truncated,
            innovation_window=iw, filter_state=fs,
            constraint_violation=info.get("constraint_violation", 0.0),
            task_id=task_id,
        ))
        obs = next_obs
        if truncated:
            break
    return transitions


def evaluate(env, agent, n_episodes=10):
    all_nees, all_cons = [], []
    for _ in range(n_episodes):
        obs, info = env.reset()
        ep_nees = []
        done = False
        while not done:
            iw = env.ekf.get_innovation_window(env.innovation_window)
            fs = np.array([
                env.ekf.state.nees, env.ekf.state.nis,
                np.trace(env.ekf.state.P), *np.diag(env.ekf.state.S),
            ])
            action = agent.select_action(obs, iw, fs, deterministic=True)
            obs, _, done, truncated, info = env.step(action)
            ep_nees.append(info.get("nees", 0.0))
            if truncated:
                break
        nees_arr = np.array(ep_nees)
        all_nees.append(np.mean(nees_arr))
        consistent = np.mean(
            (nees_arr >= env.ekf.chi2_lb) & (nees_arr <= env.ekf.chi2_ub)
        )
        all_cons.append(consistent)
    return {
        "mean_nees": float(np.mean(all_nees)),
        "consistency": float(np.mean(all_cons)),
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/train_config.yaml")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

if __name__ == "__main__":
    main()
