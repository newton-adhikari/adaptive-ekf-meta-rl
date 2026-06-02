"""
Lightweight EKF Gym environment for fast RL training.
will use numpy here for speed.
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from typing import Optional

from meta_rl.utils.ekf import EKF
from meta_rl.envs.task_sampler import TaskSampler, TaskConfig


class LightweightEKFEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, config: Optional[dict] = None):
        super().__init__()
        config = config or {}

        self.state_dim = config.get("state_dim", 6)
        self.meas_dim = config.get("meas_dim", 2)
        self.dt = config.get("dt", 0.1)
        self.episode_length = config.get("episode_length", 100)
        self.innovation_window = config.get("innovation_window", 16)

        self.n_q = config.get("n_q", self.state_dim)
        self.n_r = config.get("n_r", self.meas_dim)

        # Action: unbounded, mapped to alpha via exp() in step()
        # created this to let the agent reach alpha=0.01 (log=-4.6) to alpha=100 (log=4.6)
        self.action_space = spaces.Box(
            low=-5.0, high=5.0,
            shape=(self.n_q + self.n_r,),
            dtype=np.float32,
        )

        # Observation: log-scaled filter diagnostics
        # [last 5 innovations (flat), NEES/n, NIS/m, log(tr(P)),
        #  log(diag(P)), log(diag(Q_ekf)), log(diag(R_ekf))]
        self.n_innov = 5
        obs_dim = (
            self.n_innov * self.meas_dim   # last 5 innovations
            + 1                             # NEES / state_dim
            + 1                             # NIS / meas_dim
            + 1                             # log(1 + tr(P))
            + self.state_dim                # log(1 + diag(P))
            + self.n_q                      # log(1 + diag(Q))
            + self.n_r                      # log(1 + diag(R))
        )
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32,
        )

        self.Q_nominal = np.eye(self.state_dim) * config.get("q_nominal", 0.1)
        self.R_nominal = np.eye(self.meas_dim) * config.get("r_nominal", 1.0)

        self.task_sampler = TaskSampler(config.get("task_sampler", {}))
        self.ekf = EKF(self.state_dim, self.meas_dim, self.dt)

        self._step_count = 0
        self._x_true = None
        self._trajectory = None
        self._controls = None
        self._task: Optional[TaskConfig] = None
        self._innovations = []

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self._task = self.task_sampler.sample(self.np_random)
        self._trajectory, self._controls = self._generate_trajectory()
        self._x_true = self._trajectory[0].copy()

        x0 = self._x_true + self.np_random.normal(0, 0.1, self.state_dim)
        P0 = np.eye(self.state_dim) * 0.5
        self.ekf.reset(x0, P0, self.Q_nominal.copy(), self.R_nominal.copy())

        self._step_count = 0
        self._innovations = [[0.0] * self.meas_dim] * self.n_innov
        return self._get_obs(), self._get_info()

    def step(self, action):
        # Map unbounded action to positive alpha via exp, clamp for safety
        action = np.clip(action, -5.0, 5.0)
        alphas = np.exp(action)
        alphas = np.clip(alphas, 0.01, 100.0)

        q_alphas = alphas[: self.n_q]
        r_alphas = alphas[self.n_q :]

        Q_new = np.diag(q_alphas) @ self.Q_nominal
        R_new = np.diag(r_alphas) @ self.R_nominal
        self.ekf.set_noise(Q_new, R_new)

        # True noise (may be time-varying)
        Q_true, R_true = self._task.get_noise(self._step_count)

        # Propagate true state
        u = self._controls[min(self._step_count, len(self._controls) - 1)]
        self._x_true = self._true_dynamics(self._x_true, u, Q_true)

        # Noisy measurement
        z = self._true_measurement(self._x_true, R_true)

        # EKF step
        self.ekf.predict(u)
        nu = self.ekf.update(z)
        nees = self.ekf.compute_nees(self._x_true)

        self._innovations.append(nu.tolist())
        self._innovations = self._innovations[-self.n_innov:]

        self._step_count += 1
        done = self._step_count >= self.episode_length

        #  Reward: NEES-based (from Phase 0) 
        nees_error = abs(nees - self.state_dim)
        rmse = float(np.sqrt(np.mean((self._x_true - self.ekf.state.x) ** 2)))
        reward = -np.log1p(nees_error) - 0.1 * min(rmse, 10.0)

        # Bonus for being within chi-squared bounds
        if self.ekf.is_consistent():
            reward += 0.5

        obs = self._get_obs()
        info = self._get_info()
        info["nees"] = nees
        info["rmse"] = rmse
        info["constraint_violation"] = float(not self.ekf.is_consistent())

        return obs, reward, done, False, info

    def _get_obs(self) -> np.ndarray:
        """Log-scaled observation for numerical stability."""
        innov_flat = np.array(self._innovations).flatten()

        nees_norm = self.ekf.state.nees / max(self.state_dim, 1)
        nis_norm = self.ekf.state.nis / max(self.meas_dim, 1)
        log_tr_P = np.log1p(max(np.trace(self.ekf.state.P), 0))
        log_diag_P = np.log1p(np.maximum(np.diag(self.ekf.state.P), 0))
        log_diag_Q = np.log1p(np.maximum(np.diag(self.ekf.state.Q), 0))
        log_diag_R = np.log1p(np.maximum(np.diag(self.ekf.state.R), 0))

        obs = np.concatenate([
            innov_flat,
            [nees_norm, nis_norm, log_tr_P],
            log_diag_P,
            log_diag_Q[:self.n_q],
            log_diag_R[:self.n_r],
        ])
        return np.clip(obs, -20, 20).astype(np.float32)

    def _get_info(self) -> dict:
        return {
            "step": self._step_count,
            "nees": self.ekf.state.nees,
            "nis": self.ekf.state.nis,
            "tr_P": float(np.trace(self.ekf.state.P)),
        }

    def _generate_trajectory(self):
        T = self.episode_length + 1
        traj_type = self._task.trajectory_type
        x = np.zeros((T, self.state_dim))
        controls = np.zeros((T, 3))

        if traj_type == "circle":
            radius, omega = 2.0, 0.5
            for t in range(T):
                a = omega * t * self.dt
                x[t] = [radius*np.cos(a), radius*np.sin(a), a+np.pi/2,
                         -radius*omega*np.sin(a), radius*omega*np.cos(a), omega]
        elif traj_type == "figure8":
            s, omega = 2.0, 0.3
            for t in range(T):
                a = omega * t * self.dt
                x[t, 0] = s * np.sin(a)
                x[t, 1] = s * np.sin(2*a) / 2
                x[t, 2] = np.arctan2(s*omega*np.cos(2*a), s*omega*np.cos(a))
                x[t, 3] = s * omega * np.cos(a)
                x[t, 4] = s * omega * np.cos(2*a)
        elif traj_type == "straight":
            for t in range(T):
                x[t] = [t*self.dt, 0, 0, 1, 0, 0]
        else:
            for t in range(1, T):
                controls[t] = self.np_random.normal(0, 0.5, 3)
                x[t, 3:] = x[t-1, 3:] + controls[t] * self.dt
                x[t, :3] = x[t-1, :3] + x[t, 3:] * self.dt
        return x, controls

    def _true_dynamics(self, x, u, Q_true):
        ekf_temp = EKF(self.state_dim, self.meas_dim, self.dt)
        x_next = ekf_temp._motion_model(x, u)
        noise = self.np_random.multivariate_normal(np.zeros(self.state_dim), Q_true)
        return x_next + noise

    def _true_measurement(self, x_true, R_true):
        z_clean = x_true[:self.meas_dim]
        noise = self.np_random.multivariate_normal(np.zeros(self.meas_dim), R_true)
        return z_clean + noise
