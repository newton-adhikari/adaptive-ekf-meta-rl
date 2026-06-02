"""
Created ROS2/Gazebo-based EKF environment for sim-to-real validation.
Wraps a Gazebo simulation with ROS2 topics as a Gym environment.
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from typing import Optional

try:
    import rclpy
    from rclpy.node import Node
    from geometry_msgs.msg import Twist, PoseStamped
    from sensor_msgs.msg import Imu, LaserScan
    from nav_msgs.msg import Odometry

    ROS2_AVAILABLE = True
except ImportError:
    ROS2_AVAILABLE = False

from meta_rl.utils.ekf import EKF


class ROS2EKFEnv(gym.Env):
    """Gazebo-based EKF environment for validation.

    Subscribes to simulated sensor topics, runs EKF, and exposes
    the same Gym interface as LightweightEKFEnv for policy evaluation.

    This is used for Gazebo validation and sim-to-real evaluation.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, config: Optional[dict] = None):
        super().__init__()
        # just check this, although it will be available most of times
        if not ROS2_AVAILABLE:
            raise RuntimeError(
                "ROS2 not available. Install ROS2 Humble and source the workspace."
            )

        config = config or {}
        self.state_dim = config.get("state_dim", 6)
        self.meas_dim = config.get("meas_dim", 2)
        self.dt = config.get("dt", 0.1)
        self.episode_length = config.get("episode_length", 200)
        self.innovation_window = config.get("innovation_window", 32)
        self.n_q = config.get("n_q", self.state_dim)
        self.n_r = config.get("n_r", self.meas_dim)

        self.action_space = spaces.Box(
            low=0.1, high=10.0,
            shape=(self.n_q + self.n_r,),
            dtype=np.float32,
        )

        obs_dim = (
            self.innovation_window * self.meas_dim + 1 + 1 + 1 + self.meas_dim
        )
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32,
        )

        self.Q_nominal = np.eye(self.state_dim) * config.get("q_nominal", 0.1)
        self.R_nominal = np.eye(self.meas_dim) * config.get("r_nominal", 1.0)

        self.ekf = EKF(self.state_dim, self.meas_dim, self.dt)

        # ROS2 setup
        self._node: Optional[Node] = None
        self._latest_imu = None
        self._latest_odom = None
        self._latest_scan = None
        self._ground_truth = None

    def _init_ros2(self):
        """Initialize ROS2 node and subscribers."""
        if not rclpy.ok():
            rclpy.init()

        self._node = rclpy.create_node("ekf_env_node")

        self._node.create_subscription(
            Imu, "/imu/data", self._imu_callback, 10
        )
        self._node.create_subscription(
            Odometry, "/odom", self._odom_callback, 10
        )
        self._node.create_subscription(
            Odometry, "/ground_truth/odom", self._gt_callback, 10
        )
        self._cmd_pub = self._node.create_publisher(Twist, "/cmd_vel", 10)

    def _imu_callback(self, msg):
        self._latest_imu = msg

    def _odom_callback(self, msg):
        self._latest_odom = msg

    def _gt_callback(self, msg):
        self._ground_truth = msg

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        if self._node is None:
            self._init_ros2()

        # Reset Gazebo simulation (via service call)
        self._reset_simulation()

        x0 = np.zeros(self.state_dim)
        P0 = np.eye(self.state_dim) * 0.5
        self.ekf.reset(x0, P0, self.Q_nominal.copy(), self.R_nominal.copy())

        self._step_count = 0
        obs = self._get_obs()
        return obs, {}

    def step(self, action):
        action = np.clip(action, self.action_space.low, self.action_space.high)

        # Apply Q/R scaling
        q_scales = action[: self.n_q]
        r_scales = action[self.n_q :]
        Q_new = np.diag(q_scales) @ self.Q_nominal[: self.n_q, : self.n_q]
        R_new = np.diag(r_scales) @ self.R_nominal[: self.n_r, : self.n_r]
        self.ekf.set_noise(Q_new, R_new)

        # Spin ROS2 to get latest sensor data
        rclpy.spin_once(self._node, timeout_sec=self.dt)

        # Extract measurement from odometry
        z = self._extract_measurement()
        u = self._extract_control()

        # EKF step
        self.ekf.predict(u)
        self.ekf.update(z)

        # Compute NEES 
        nees = 0.0
        if self._ground_truth is not None:
            x_true = self._extract_ground_truth()
            nees = self.ekf.compute_nees(x_true)

        self._step_count += 1
        done = self._step_count >= self.episode_length

        error = self._extract_ground_truth() - self.ekf.state.x if self._ground_truth else np.zeros(self.state_dim)
        reward = -float(np.sum(error ** 2))

        obs = self._get_obs()
        info = {
            "nees": nees,
            "nis": self.ekf.state.nis,
            "constraint_violation": float(not self.ekf.is_consistent()),
        }
        return obs, reward, done, False, info

    def _get_obs(self) -> np.ndarray:
        innov_window = self.ekf.get_innovation_window(self.innovation_window)
        innov_flat = innov_window.flatten()
        nees = np.array([self.ekf.state.nees])
        nis = np.array([self.ekf.state.nis])
        tr_P = np.array([np.trace(self.ekf.state.P)])
        diag_S = np.diag(self.ekf.state.S)
        return np.concatenate([innov_flat, nees, nis, tr_P, diag_S]).astype(np.float32)

    def _extract_measurement(self) -> np.ndarray:
        if self._latest_odom is not None:
            pos = self._latest_odom.pose.pose.position
            return np.array([pos.x, pos.y])
        return np.zeros(self.meas_dim)

    def _extract_control(self) -> np.ndarray:
        if self._latest_imu is not None:
            acc = self._latest_imu.linear_acceleration
            ang = self._latest_imu.angular_velocity
            return np.array([acc.x, acc.y, ang.z])
        return np.zeros(3)

    def _extract_ground_truth(self) -> np.ndarray:
        if self._ground_truth is not None:
            pos = self._ground_truth.pose.pose.position
            vel = self._ground_truth.twist.twist.linear
            ang = self._ground_truth.twist.twist.angular
            
            # Extract yaw from quaternion [same ole]
            q = self._ground_truth.pose.pose.orientation
            yaw = np.arctan2(
                2.0 * (q.w * q.z + q.x * q.y),
                1.0 - 2.0 * (q.y ** 2 + q.z ** 2),
            )
            return np.array([pos.x, pos.y, yaw, vel.x, vel.y, ang.z])
        return np.zeros(self.state_dim)

    def _reset_simulation(self):
        """Reset Gazebo simulation via ROS2 service."""
        # Placeholder using, requires gazebo_msgs/srv/ResetSimulation
        pass

    def close(self):
        if self._node is not None:
            self._node.destroy_node()
            self._node = None
