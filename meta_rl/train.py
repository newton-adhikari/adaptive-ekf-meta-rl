"""
CC-MetaEKF training script.


python -m meta_rl.train --config configs/train_config.yaml --seed 42
"""

import argparse
import os
import time
import yaml
import numpy as np
import torch
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", message=".*NNPACK.*")

from meta_rl.envs.lightweight_ekf_env import LightweightEKFEnv
from meta_rl.agents.pid_ccpo_agent import PIDCCPOAgent
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
    # Set up command-line arguments so we can easily swap configs or seeds from the terminal
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, default="configs/train_config.yaml"
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Load our settings file and lock down the random seeds so results are reproducible
    config = load_config(args.config)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Max out available CPU threads good when switching from GPU to CPU
    n_cpus = os.cpu_count() or 1
    torch.set_num_threads(n_cpus)
    print(f"Using {n_cpus} CPU threads", flush=True)

    # load the Extended Kalman Filter (EKF) environment using our configuration setup
    env_config = config.get("environment", {})
    env = LightweightEKFEnv(env_config)

    # Finding dimensions of our states and actions to properly build the neural net
    agent_config = config.get("agent", {})
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    meas_dim = env_config.get("meas_dim", 2)

    # start the main agent 
    agent = PIDCCPOAgent(
        state_dim=obs_dim,
        action_dim=act_dim,
        innovation_dim=meas_dim,
        filter_state_dim=1 + 1 + 1 + meas_dim,
        config=agent_config,
    )

    # Set up a multi-task replay buffer to store and manage our training experiences
    train_config = config.get("training", {})
    buffer = MultiTaskReplayBuffer(
        capacity=train_config.get("buffer_capacity", 50_000),
        max_tasks=train_config.get("buffer_max_tasks", 200),
    )

    # checkpoints directory if it doesn't already exist
    ckpt_dir = Path("checkpoints")
    ckpt_dir.mkdir(exist_ok=True)

    # hyperparameters for the training loop
    n_epochs = train_config.get("n_epochs", 200)
    tasks_per_epoch = train_config.get("tasks_per_epoch", 4)
    context_episodes = train_config.get("context_episodes", 1)
    train_episodes = train_config.get("train_episodes", 2)
    gradient_steps = train_config.get("gradient_steps", 20)
    eval_interval = train_config.get("eval_interval", 10)
    save_interval = train_config.get("save_interval", 50)
    batch_size = agent_config.get("batch_size", 256)

    # tracking the highest consistency score to safely save our absolute best model
    best_consistency = 0.0
    eps_per_epoch = tasks_per_epoch * (context_episodes + train_episodes)

    print(
        f"Training: {n_epochs} epochs, {tasks_per_epoch} tasks/epoch, "
        f"{eps_per_epoch} episodes/epoch",
        flush=True,
    )

    # The core training loop starts
    for epoch in range(1, n_epochs + 1):
        t0 = time.time()

        # Phase 1: Interaction & Data Collection
        for task_id in range(tasks_per_epoch):
            for _ in range(context_episodes + train_episodes):
                transitions = collect_episode(env, agent, task_id)
                buffer.add_episode(transitions)

                # Extracting states and rewards to dynamically update our feature normalizers
                obs_batch = np.array([t.state for t in transitions])
                rew_batch = np.array([t.reward for t in transitions]).reshape(
                    -1, 1
                )
                agent.obs_normalizer.update(obs_batch)
                agent.reward_normalizer.update(rew_batch * agent.reward_scale)

        collect_t = time.time() - t0
        t1 = time.time()

        # Phase 2: Neural Network Optimization
        if buffer.size >= batch_size:
            epoch_losses = []
            for _ in range(gradient_steps):
                # find a random batch of experience data and optimize network weights
                batch = buffer.sample(batch_size, device=agent.device)
                losses = agent.update(batch)
                epoch_losses.append(losses)

            update_t = time.time() - t1

            # Average the losses from this epoch and print progress update to the console
            avg = {
                k: np.mean([l[k] for l in epoch_losses]) for k in epoch_losses[0]
            }
            print(
                f"\rEpoch {epoch}/{n_epochs} | "
                f"q={avg['q_loss']:.2f} | pi={avg['policy_loss']:.2f} | "
                f"lam={avg['lambda']:.4f} | viol={avg['avg_violation']:.3f} | "
                f"{collect_t:.0f}s+{update_t:.0f}s",
                flush=True,
            )

        # Phase 3: Periodic Evaluation
        if epoch % eval_interval == 0:
            m = evaluate(env, agent)
            tag = ""

            # Check this evaluation cycle better than our previous record for consistency
            if m["consistency"] > best_consistency:
                best_consistency = m["consistency"]
                agent.save(str(ckpt_dir / "best_model.pt"))
                tag = " *"  # pointed with an asterisk for visual in logs

            # Output current filter metrics NEES and consistency percentages
            print(
                f"  [Eval] NEES={m['mean_nees']:.1f} | "
                f"Consistency={m['consistency']:.1%}{tag}",
                flush=True,
            )

        # lost data due to overheating, so done Routine Checkpointing
        if epoch % save_interval == 0:
            agent.save(str(ckpt_dir / f"checkpoint_{epoch}.pt"))

    #  Save the absolute final state of the agent
    agent.save(str(ckpt_dir / "final_model.pt"))
    print(f"Done. Best consistency: {best_consistency:.1%}", flush=True)


if __name__ == "__main__":
    main()
