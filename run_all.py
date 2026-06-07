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