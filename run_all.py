#!/usr/bin/env python3
"""
CC-MetaEKF: One-command training pipeline.

runs all experiments, saves results + logs.

"""
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal
import time, argparse, json, os, sys, warnings, logging
from pathlib import Path
from datetime import datetime

warnings.filterwarnings("ignore")


# ================================================================
# Setup
# ================================================================

def setup_device():
    if torch.cuda.is_available():
        dev = "cuda"
        name = torch.cuda.get_device_name()
        torch.backends.cudnn.benchmark = True
    else:
        dev = "cpu"
        name = f"{os.cpu_count()} CPU cores"
        torch.set_num_threads(os.cpu_count() or 1)
    return dev, name

DEVICE, DEVICE_NAME = setup_device()

def setup_logging(output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = output_dir / f"run_{ts}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file),
        ],
    )
    return logging.getLogger("ccmetaekf"), log_file

def log_and_save(results, name, output_dir):
    path = output_dir / f"{name}.json"
    with open(path, "w") as f:
        json.dump(results, f, indent=2, default=float)
    return path


# ================================================================
# Extended Kalman Filter (EKF)
# ================================================================

class EKF6D:
    def __init__(self, dt=0.1):
        # Setting filter dimensions:
        # n = number of states
        # m = number of measurements
        self.n = 6
        self.m = 2
        self.dt = dt

        # Storing lower and upper Chi-square bounds
        # for checking filter consistency using NEES
        self.chi2_lb = 1.237
        self.chi2_ub = 14.449

        # Initializing performance metrics
        self.nees = 0.0
        self.nis = 0.0

        # Initializing innovation covariance matrix
        self.S = np.eye(2)

    def reset(self, x0, P0, Q, R):
        # Loading initial state estimate
        self.x = x0.copy().astype(np.float64)

        # Loading initial covariance matrix
        self.P = P0.copy().astype(np.float64)

        # Storing process noise covariance
        self.Q = Q.copy().astype(np.float64)

        # Storing measurement noise covariance
        self.R = R.copy().astype(np.float64)

        # Resetting consistency metrics
        self.nees = 0.0
        self.nis = 0.0

        # Resetting innovation covariance
        self.S = np.eye(2)

    def _F(self, x):
        # Computing Jacobian matrix of the motion model
        _, _, th, vx, vy, _ = x
        dt = self.dt

        F = np.eye(6)

        # Calculating position sensitivity to heading
        F[0,2] = (-vx*np.sin(th) - vy*np.cos(th)) * dt
        F[0,3] = np.cos(th) * dt
        F[0,4] = -np.sin(th) * dt

        F[1,2] = (vx*np.cos(th) - vy*np.sin(th)) * dt
        F[1,3] = np.sin(th) * dt
        F[1,4] = np.cos(th) * dt

        # Relating heading angle to angular velocity
        F[2,5] = dt

        return F

    def _f(self, x, u):
        # Predicting next state using the nonlinear motion model
        px, py, th, vx, vy, om = x
        dt = self.dt

        return np.array([
            px + vx*np.cos(th)*dt - vy*np.sin(th)*dt,  # Updating x position
            py + vx*np.sin(th)*dt + vy*np.cos(th)*dt,  # Updating y position
            th + om*dt,                                # Updating heading angle
            vx + u[0]*dt,                              # Updating x velocity
            vy + u[1]*dt,                              # Updating y velocity
            om + u[2]*dt                               # Updating angular velocity
        ])

    def predict(self, u):
        # Computing Jacobian at current state
        F = self._F(self.x)

        # Predicting next state estimate
        self.x = self._f(self.x, u)

        # Predicting covariance growth
        self.P = F @ self.P @ F.T + self.Q

        # Keeping covariance symmetric
        self.P = (self.P + self.P.T) / 2

    def update(self, z):
        # Creating measurement matrix
        # Measuring only x and y positions
        H = np.zeros((2,6))
        H[0,0] = 1
        H[1,1] = 1

        # Computing innovation (measurement error)
        nu = z - H @ self.x

        # Computing innovation covariance
        S = H @ self.P @ H.T + self.R

        # Computing Kalman gain
        K = self.P @ H.T @ np.linalg.inv(S)

        # Updating state estimate
        self.x = self.x + K @ nu

        # Updating covariance using Joseph form
        I_KH = np.eye(6) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ self.R @ K.T

        # Keeping covariance symmetric
        self.P = (self.P + self.P.T) / 2

        # Storing innovation covariance
        self.S = S

        # Calculating NIS for measurement consistency checking
        self.nis = float(nu @ np.linalg.inv(S) @ nu)

        return nu

    def compute_nees(self, x_true):
        # Computing estimation error
        e = x_true - self.x

        try:
            # Calculating NEES for state consistency checking
            self.nees = float(e @ np.linalg.inv(self.P) @ e)
        except:
            # Using large penalty value if covariance inversion fails
            self.nees = 100.0

        return self.nees

    def is_consistent(self):
        # Checking whether NEES lies inside expected bounds
        return self.chi2_lb <= self.nees <= self.chi2_ub


# ================================================================
# 1D EKF Used During Phase 0
# Tracking position and velocity in one dimension
# ================================================================

class EKF1D:
    def __init__(self, dt=0.1):
        self.dt = dt

        # Defining constant velocity state transition model
        self.F = np.array([
            [1.0, dt],
            [0.0, 1.0]
        ])

        # Defining measurement model
        # Measuring position only
        self.H = np.array([
            [1.0, 0.0]
        ])

    def reset(self, x0, P0, Q, R):
        # Loading initial state estimate
        self.x = x0.copy().astype(np.float64)

        # Loading initial covariance estimate
        self.P = P0.copy().astype(np.float64)

        # Storing process noise covariance
        self.Q = Q.copy().astype(np.float64)

        # Storing measurement noise covariance
        self.R = R.copy().astype(np.float64)

    def predict(self):
        # Predicting next state
        self.x = self.F @ self.x

        # Predicting covariance growth
        self.P = self.F @ self.P @ self.F.T + self.Q

        # Keeping covariance symmetric
        self.P = (self.P + self.P.T) / 2

    def update(self, z):
        # Computing measurement residual
        nu = z - self.H @ self.x

        # Computing residual covariance
        S = self.H @ self.P @ self.H.T + self.R

        # Computing Kalman gain
        K = (self.P @ self.H.T) / S[0,0]

        # Updating state estimate
        self.x = self.x + K[:,0] * nu[0]

        # Updating covariance using Joseph form
        I_KH = np.eye(2) - K @ self.H.reshape(1,2)
        self.P = I_KH @ self.P @ I_KH.T + K @ self.R @ K.T

        # Keeping covariance symmetric
        self.P = (self.P + self.P.T) / 2

        return nu[0], S[0,0]

    def nees(self, x_true):
        # Computing estimation error
        e = x_true - self.x

        try:
            # Calculating NEES for filter consistency evaluation
            return float(e @ np.linalg.inv(self.P) @ e)
        except:
            # Returning large value if inversion fails
            return 100.0
        

# ================================================================
# Tasks and Environments
# ================================================================

class Task:
    def __init__(self, Q_diag, R_diag, regime="stationary", ct=50):

        # Stored baseline process and measurement noise
        self.Q_base = np.array(Q_diag)
        self.R_base = np.array(R_diag)

        # Remembered what kind of chaos this task wanted
        self.regime = regime
        self.change_time = ct

        # Generated future noise values for abrupt changes
        # Basically preparing a surprise attack for the EKF
        self.Q_after = self.Q_base * np.random.choice([0.2, 5.0], size=len(Q_diag))
        self.R_after = self.R_base * np.random.choice([0.2, 5.0], size=len(R_diag))

    def get_noise(self, t):

        # Keeping life easy: noise never changed
        if self.regime == "stationary":
            return np.diag(self.Q_base), np.diag(self.R_base)

        # Noise suddenly woke up and chose violence
        elif self.regime == "abrupt" and t >= self.change_time:
            return np.diag(self.Q_after), np.diag(self.R_after)

        # Slowly increasing uncertainty over time
        elif self.regime == "drift":
            s = 1 + 0.005 * t
            return np.diag(self.Q_base * s), np.diag(self.R_base * s)

        # Falling back to default noise values
        return np.diag(self.Q_base), np.diag(self.R_base)


def sample_task(rng):

    # Sampled process noise from a log-uniform distribution
    q = np.exp(rng.uniform(np.log(0.005), np.log(0.2), 6))

    # Sampled measurement noise
    r = np.exp(rng.uniform(np.log(0.05), np.log(2.0), 2))

    # Randomly picked a world behavior
    regime = rng.choice(
        ["stationary", "abrupt", "drift"],
        p=[0.5, 0.3, 0.2]
    )

    # Created a fresh task for the agent to suffer through
    return Task(q, r, regime, rng.integers(30, 70))


class Env6D:

    def __init__(self, ep_len=100):

        # Stored episode length
        self.ep_len = ep_len

        # Created EKF instance
        self.ekf = EKF6D()

        # Set nominal covariance values
        self.Q_nom = np.eye(6) * 0.05
        self.R_nom = np.eye(2) * 0.5

        # Keeping recent innovation history
        self.ctx_len = 30

        # Number of innovations exposed to the policy
        self.n_innov = 5

        # Observation vector size
        self.obs_dim = self.n_innov*2 + 1 + 1 + 1 + 6 + 6 + 2

        # Action controls 6 Q values + 2 R values
        self.act_dim = 8

        # Created RNG for generating random nonsense responsibly
        self.rng = np.random.default_rng()

    def reset(self, task=None):

        # Generated a new task if none was supplied
        self.task = task or sample_task(self.rng)

        # Reset episode clock
        self.t = 0

        # Chose a trajectory type
        traj = self.rng.choice(["circle", "straight", "random"])
        self.traj_type = traj

        # Started with zero state
        self.x_true = np.zeros(6)

        # Spawned circular motion
        if traj == "circle":
            self.x_true = np.array([2, 0, np.pi/2, 0, 1.0, 0.5])

        # Spawned boring but predictable straight motion
        elif traj == "straight":
            self.x_true = np.array([0, 0, 0, 1, 0, 0])

        # Added initialization error because perfect sensors only exist in slides
        x0 = self.x_true + self.rng.normal(0, 0.2, 6)

        # Reset EKF state
        self.ekf.reset(
            x0,
            np.eye(6) * 0.5,
            self.Q_nom.copy(),
            self.R_nom.copy()
        )

        # Started innovation history
        self.innovs = [[0.0, 0.0]] * self.ctx_len

        # Started NEES history tracking
        self.nees_history = []

        return self._obs(), self._ctx()

    def step(self, action):

        # Converted actions into positive scaling factors
        alphas = np.clip(np.exp(action), 0.01, 100.0)

        # Updated EKF covariance guesses
        self.ekf.Q = np.diag(alphas[:6]) @ self.Q_nom
        self.ekf.R = np.diag(alphas[6:]) @ self.R_nom

        # Retrieved actual environment noise
        Q_true, R_true = self.task.get_noise(self.t)

        # Started with zero control input
        u = np.zeros(3)

        # Generated random controls for the random trajectory
        if self.traj_type == "random":
            u = self.rng.normal(0, 0.3, 3)

        # Propagated true state using real noise
        self.x_true = (
            self.ekf._f(self.x_true, u)
            + self.rng.multivariate_normal(np.zeros(6), Q_true)
        )

        # Generated noisy position measurement
        z = (
            self.x_true[:2]
            + self.rng.multivariate_normal(np.zeros(2), R_true)
        )

        # Running EKF predict-update cycle
        self.ekf.predict(u)
        nu = self.ekf.update(z)

        # Calculated consistency score
        nees = self.ekf.compute_nees(self.x_true)

        # Stored latest innovation
        self.innovs.append(nu.tolist())

        # Keeping only recent history
        self.innovs = self.innovs[-self.ctx_len:]

        self.t += 1

        # Checking whether episode finished
        done = self.t >= self.ep_len

        # Tracking NEES history for smoother rewards
        self.nees_history.append(nees)

        # Averaging recent consistency values
        avg_nees = np.mean(self.nees_history[-20:])

        # Measuring state estimation error
        rmse = float(
            np.sqrt(np.mean((self.x_true - self.ekf.x) ** 2))
        )

        # Rewarding NEES staying near target value
        # Less drama = more reward
        reward = -np.log1p(abs(avg_nees - 6))

        # Punishing large estimation errors
        reward -= 0.05 * min(rmse, 10.0)

        # EKF behaving itself earned bonus points
        if self.ekf.is_consistent():
            reward += 0.3

        # Running average staying in healthy range
        if 3.0 <= avg_nees <= 12.0:
            reward += 0.3

        return (
            self._obs(),
            self._ctx(),
            reward,
            done,
            {
                "nees": nees,
                "rmse": rmse,
                "consistent": self.ekf.is_consistent()
            }
        )

    def _obs(self):

        # Flattening recent innovations into one vector
        iv = np.array(
            self.innovs[-self.n_innov:]
        ).flatten()

        # Building observation features for the policy
        # Compressing huge values before they start causing drama
        return np.clip(
            np.concatenate([
                iv,

                # Normalized consistency metrics
                [self.ekf.nees / 6,
                 self.ekf.nis / 2],

                # Tracking covariance growth
                [np.log1p(max(np.trace(self.ekf.P), 0))],

                # State uncertainty
                np.log1p(np.maximum(np.diag(self.ekf.P), 0)),

                # Process noise estimate
                np.log1p(np.maximum(np.diag(self.ekf.Q), 0)),

                # Measurement noise estimate
                np.log1p(np.maximum(np.diag(self.ekf.R), 0))
            ]),
            -20,
            20
        ).astype(np.float32)

    def _ctx(self):

        # Returning full innovation history
        # Agent memory, because forgetting is bad
        return np.array(
            self.innovs,
            dtype=np.float32
        ).flatten()


