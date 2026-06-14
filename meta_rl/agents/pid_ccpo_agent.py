"""
The main contribution: 
    PID-Lagrangian Consistency-Constrained Policy Optimization (PID-CCPO).

This enforces NEES consistency as a hard constraint via
PID-controlled Lagrangian relaxation rather than fragile reward shaping.

This combines:
  - SAC (Soft Actor-Critic) for the base RL objective.
  - PID-Lagrangian dual variable for the consistency constraint.
  - Cost critic for predicting constraint violations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from copy import deepcopy
from typing import Optional

from meta_rl.agents.st_sie_encoder import STSIEEncoder
from meta_rl.agents.raw_encoder import RawInnovationEncoder
from meta_rl.agents.policy_network import GaussianPolicy, QNetwork
from meta_rl.agents.replay_buffer import Batch


class RunningNormalizer:
    """
    Online running mean/variance using Welford's algorithm.

    this will normalize inputs to approximately zero mean and unit variance.
    """

    def __init__(self, shape: tuple, eps: float = 1e-8):
        self.shape = shape
        self.eps = eps
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = 0

    def update(self, x: np.ndarray):
        """Update running statistics with a batch of data.

        """
        x = np.asarray(x, dtype=np.float64)
        if x.ndim == len(self.shape):
            x = x[None]  # add batch dim

        batch_mean = x.mean(axis=0)
        batch_var = x.var(axis=0)
        batch_count = x.shape[0]

        self._update_from_moments(batch_mean, batch_var, batch_count)

    def _update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        total = self.count + batch_count

        new_mean = self.mean + delta * batch_count / max(total, 1)
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + delta ** 2 * self.count * batch_count / max(total, 1)
        new_var = m2 / max(total, 1)

        self.mean = new_mean
        self.var = np.maximum(new_var, self.eps)
        self.count = total

    def normalize(self, x):
        """
        the function which normalizes input using running statistics.

        Works with both numpy arrays and torch tensors.
        """
        if isinstance(x, torch.Tensor):
            mean = torch.tensor(self.mean, dtype=x.dtype, device=x.device)
            std = torch.tensor(np.sqrt(self.var) + self.eps, dtype=x.dtype, device=x.device)
            return (x - mean) / std
        else:
            return (x - self.mean) / (np.sqrt(self.var) + self.eps)

    def state_dict(self) -> dict:
        return {"mean": self.mean.copy(), "var": self.var.copy(), "count": self.count}

    def load_state_dict(self, state: dict):
        self.mean = state["mean"].copy()
        self.var = state["var"].copy()
        self.count = state["count"]


class PIDLagrangian:
    """This is PID controller for the Lagrangian dual variable λ.

    Updates λ based on constraint violation error using PID control,
    which is more stable than vanilla gradient-based dual updates.
    """

    def __init__(
        self,
        delta: float = 0.1,
        k_p: float = 0.1,
        k_i: float = 0.01,
        k_d: float = 0.01,
        integral_max: float = 10.0,
    ):
        self.delta = delta
        self.k_p = k_p
        self.k_i = k_i
        self.k_d = k_d
        self.integral_max = integral_max

        self._lambda = 0.0
        self._integral = 0.0
        self._prev_error = 0.0

    def update(self, avg_violation: float) -> float:
        """
        this updates λ based on current average constraint violation.
        """
        error = avg_violation - self.delta
        self._integral += error
        # Clamp integral to prevent windup
        self._integral = max(-self.integral_max, min(self.integral_max, self._integral))

        derivative = error - self._prev_error
        self._prev_error = error

        self._lambda = max(
            0.0,
            self.k_p * error + self.k_i * self._integral + self.k_d * derivative,
        )
        return self._lambda

    @property
    def value(self) -> float:
        return self._lambda

    def reset(self):
        self._lambda = 0.0
        self._integral = 0.0
        self._prev_error = 0.0



class PIDCCPOAgent:
    """
    This is the CC-MetaEKF agent: 
        SAC + PID-Lagrangian consistency constraint.
    
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        innovation_dim: int = 2,
        filter_state_dim: int = 5,
        config: Optional[dict] = None,
    ):
        config = config or {}
        self.device = config.get("device", "cpu")
        self.gamma = config.get("gamma", 0.95)
        self.tau = config.get("tau", 0.005)
        self.batch_size = config.get("batch_size", 256)
        self.reward_scale = config.get("reward_scale", 0.01)

        latent_dim = config.get("latent_dim", 32)
        hidden_dim = config.get("hidden_dim", 256)
        lr = config.get("lr", 3e-4)

        # Running normalizers for stable training
        self.obs_normalizer = RunningNormalizer(shape=(state_dim,))
        self.reward_normalizer = RunningNormalizer(shape=(1,))

        # Encoder (ST-SIE or raw GRU for ablation)
        encoder_type = config.get("encoder_type", "st_sie")
        if encoder_type == "st_sie":
            self.encoder = STSIEEncoder(
                innovation_dim=innovation_dim,
                filter_state_dim=filter_state_dim,
                latent_dim=latent_dim,
                window_size=config.get("stft_window", 32),
                hop_size=config.get("stft_hop", 8),
            ).to(self.device)
        else:
            self.encoder = RawInnovationEncoder(
                innovation_dim=innovation_dim,
                filter_state_dim=filter_state_dim,
                latent_dim=latent_dim,
            ).to(self.device)

        # Policy
        self.policy = GaussianPolicy(
            state_dim=state_dim,
            context_dim=latent_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
        ).to(self.device)

        # Twin Q-networks
        self.q1 = QNetwork(state_dim, latent_dim, action_dim, hidden_dim).to(self.device)
        self.q2 = QNetwork(state_dim, latent_dim, action_dim, hidden_dim).to(self.device)
        self.q1_target = deepcopy(self.q1)
        self.q2_target = deepcopy(self.q2)

        # Cost critic
        self.cost_critic = QNetwork(
            state_dim, latent_dim, action_dim, hidden_dim
        ).to(self.device)
        self.cost_critic_target = deepcopy(self.cost_critic)

        # SAC entropy coefficient (auto-tuned)
        self.target_entropy = -action_dim
        self.log_alpha = torch.tensor(
            np.log(0.2), dtype=torch.float32, requires_grad=True, device=self.device
        )

        # PID-Lagrangian with integral clamping
        self.pid = PIDLagrangian(
            delta=config.get("delta", 0.1),
            k_p=config.get("pid_kp", 0.1),
            k_i=config.get("pid_ki", 0.01),
            k_d=config.get("pid_kd", 0.01),
            integral_max=config.get("pid_integral_max", 10.0),
        )

        # Optimizers
        self.encoder_optimizer = torch.optim.Adam(self.encoder.parameters(), lr=lr)
        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        self.q_optimizer = torch.optim.Adam(
            list(self.q1.parameters()) + list(self.q2.parameters()), lr=lr
        )
        self.cost_optimizer = torch.optim.Adam(self.cost_critic.parameters(), lr=lr)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=lr)

        self._update_step = 0

    def select_action(
        self,
        state: np.ndarray,
        innovation_window: np.ndarray,
        filter_state: np.ndarray,
        deterministic: bool = False,
    ) -> np.ndarray:
    
        """Select action given current observation and context."""
        with torch.no_grad():
            s = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            # Normalize observation
            s = self.obs_normalizer.normalize(s)

            iw = torch.tensor(
                innovation_window, dtype=torch.float32, device=self.device
            ).unsqueeze(0)
            fs = torch.tensor(
                filter_state, dtype=torch.float32, device=self.device
            ).unsqueeze(0)

            z = self.encoder(iw, fs)

            if deterministic:
                action = self.policy.deterministic(s, z)
            else:
                action, _ = self.policy.sample(s, z)

        return action.cpu().numpy().squeeze(0)

    def update(self, batch: Batch) -> dict:
        """Perform one gradient step on all components."""
        
        #  Normalize observations and rewards 
        self.obs_normalizer.update(batch.states.cpu().numpy())
        self.reward_normalizer.update(
            (batch.rewards * self.reward_scale).cpu().numpy()
        )

        states = self.obs_normalizer.normalize(batch.states)
        next_states = self.obs_normalizer.normalize(batch.next_states)
        rewards = self.reward_normalizer.normalize(batch.rewards * self.reward_scale)

        # Encode context
        z = self.encoder(batch.innovation_windows, batch.filter_states)
        z_detached = z.detach()

        #  Critic update 
        with torch.no_grad():
            next_z = z_detached
            next_actions, next_log_probs = self.policy.sample(next_states, next_z)
            q1_next = self.q1_target(next_states, next_actions, next_z)
            q2_next = self.q2_target(next_states, next_actions, next_z)
            q_next = torch.min(q1_next, q2_next) - self.log_alpha.exp() * next_log_probs
            q_target = rewards + (1 - batch.dones) * self.gamma * q_next

        q1_pred = self.q1(states, batch.actions, z_detached)
        q2_pred = self.q2(states, batch.actions, z_detached)
        q_loss = F.mse_loss(q1_pred, q_target) + F.mse_loss(q2_pred, q_target)

        self.q_optimizer.zero_grad()
        q_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q1.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(self.q2.parameters(), 1.0)
        self.q_optimizer.step()

        #  Cost critic update 
        with torch.no_grad():
            cost_next = self.cost_critic_target(next_states, next_actions, next_z)
            cost_target = batch.constraint_violations + (
                1 - batch.dones
            ) * self.gamma * cost_next

        cost_pred = self.cost_critic(states, batch.actions, z_detached)
        cost_loss = F.mse_loss(cost_pred, cost_target)

        self.cost_optimizer.zero_grad()
        cost_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.cost_critic.parameters(), 1.0)
        self.cost_optimizer.step()

        #  Policy + Encoder update 
        z_for_policy = self.encoder(batch.innovation_windows, batch.filter_states)
        actions_new, log_probs_new = self.policy.sample(states, z_for_policy)
        q1_new = self.q1(states, actions_new, z_for_policy)
        q2_new = self.q2(states, actions_new, z_for_policy)
        q_new = torch.min(q1_new, q2_new)

        cost_new = self.cost_critic(states, actions_new, z_for_policy)

        # Normalize Q-values to stabilize policy gradient
        q_scale = max(q_new.abs().mean().item(), 1.0)
        q_normalized = q_new / q_scale

        alpha = self.log_alpha.exp().detach()
        lam = self.pid.value

        policy_loss = (
            alpha * log_probs_new - q_normalized + lam * cost_new
        ).mean()

        self.encoder_optimizer.zero_grad()
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
        self.encoder_optimizer.step()
        self.policy_optimizer.step()

        #  Alpha update 
        alpha_loss = -(
            self.log_alpha * (log_probs_new.detach() + self.target_entropy)
        ).mean()

        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        #  PID-Lagrangian update 
        avg_violation = batch.constraint_violations.mean().item()
        self.pid.update(avg_violation)

        #  Soft target updates 
        self._soft_update(self.q1, self.q1_target)
        self._soft_update(self.q2, self.q2_target)
        self._soft_update(self.cost_critic, self.cost_critic_target)

        self._update_step += 1

        return {
            "q_loss": q_loss.item(),
            "cost_loss": cost_loss.item(),
            "policy_loss": policy_loss.item(),
            "alpha_loss": alpha_loss.item(),
            "alpha": self.log_alpha.exp().item(),
            "lambda": self.pid.value,
            "avg_violation": avg_violation,
            "update_step": self._update_step,
        }

    def _soft_update(self, source: nn.Module, target: nn.Module):
        for sp, tp in zip(source.parameters(), target.parameters()):
            tp.data.copy_(self.tau * sp.data + (1 - self.tau) * tp.data)

    def save(self, path: str):
        torch.save({
            "encoder": self.encoder.state_dict(),
            "policy": self.policy.state_dict(),
            "q1": self.q1.state_dict(),
            "q2": self.q2.state_dict(),
            "cost_critic": self.cost_critic.state_dict(),
            "log_alpha": self.log_alpha.data,
            "pid_lambda": self.pid.value,
            "pid_integral": self.pid._integral,
            "update_step": self._update_step,
            "obs_normalizer": self.obs_normalizer.state_dict(),
            "reward_normalizer": self.reward_normalizer.state_dict(),
        }, path)

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.encoder.load_state_dict(ckpt["encoder"])
        self.policy.load_state_dict(ckpt["policy"])
        self.q1.load_state_dict(ckpt["q1"])
        self.q2.load_state_dict(ckpt["q2"])
        self.cost_critic.load_state_dict(ckpt["cost_critic"])
        self.log_alpha.data = ckpt["log_alpha"]
        self.pid._lambda = ckpt["pid_lambda"]
        self.pid._integral = ckpt.get("pid_integral", 0.0)
        self._update_step = ckpt["update_step"]
        if "obs_normalizer" in ckpt:
            self.obs_normalizer.load_state_dict(ckpt["obs_normalizer"])
        if "reward_normalizer" in ckpt:
            self.reward_normalizer.load_state_dict(ckpt["reward_normalizer"])
        self.q1_target = deepcopy(self.q1)
        self.q2_target = deepcopy(self.q2)
        self.cost_critic_target = deepcopy(self.cost_critic)
