"""
This is the main online CC-MetaEKF ROS2 Node.
which runs the trained meta-RL policy for real-time EKF noise adaptation.

"""

import numpy as np
import os

try:
    import torch
    import rclpy
    from rclpy.node import Node
    from geometry_msgs.msg import PoseWithCovarianceStamped
    from sensor_msgs.msg import Imu
    from nav_msgs.msg import Odometry
    from std_msgs.msg import Float64MultiArray
    ROS2_AVAILABLE = True
except ImportError:
    ROS2_AVAILABLE = False

from meta_rl.utils.ekf import EKF
from meta_rl.agents.st_sie_encoder import STSIEEncoder
from meta_rl.agents.policy_network import GaussianPolicy


if ROS2_AVAILABLE:
    class MetaEKFNode(Node):
        """Online CC-MetaEKF node with two-timescale inference."""

        def __init__(self):
            super().__init__("meta_ekf_node")

            # Parameters
            self.declare_parameter("dt", 0.01)
            self.declare_parameter("state_dim", 6)
            self.declare_parameter("meas_dim", 2)
            self.declare_parameter("checkpoint_path", "checkpoints/best_model.pt")
            self.declare_parameter("encoder_update_interval", 10)  # K
            self.declare_parameter("innovation_window", 32)

            dt = self.get_parameter("dt").value
            state_dim = self.get_parameter("state_dim").value
            meas_dim = self.get_parameter("meas_dim").value
            ckpt_path = self.get_parameter("checkpoint_path").value
            self.K = self.get_parameter("encoder_update_interval").value
            self.innov_window_size = self.get_parameter("innovation_window").value

            # EKF
            self.ekf = EKF(state_dim, meas_dim, dt)
            self.ekf.reset(
                x0=np.zeros(state_dim),
                P0=np.eye(state_dim) * 0.5,
                Q=np.eye(state_dim) * 0.1,
                R=np.eye(meas_dim) * 1.0,
            )

            # Load trained models
            self._load_models(ckpt_path, state_dim, meas_dim)

            # Cached context
            self._cached_z = torch.zeros(1, 32)
            self._step_count = 0

            # Subscribers
            self.create_subscription(Imu, "/imu/data", self.imu_callback, 10)
            self.create_subscription(Odometry, "/odom", self.odom_callback, 10)

            # Publishers
            self.state_pub = self.create_publisher(
                PoseWithCovarianceStamped, "/meta_ekf/state", 10
            )
            self.diagnostics_pub = self.create_publisher(
                Float64MultiArray, "/meta_ekf/diagnostics", 10
            )

            self.get_logger().info(
                f"Meta-EKF node initialized (K={self.K})"
            )

        def _load_models(self, ckpt_path, state_dim, meas_dim):
            """Load trained policy from checkpoint (run_all.py format).

            The checkpoint from run_all.py saves the full Policy model state_dict
            which includes encoder + actor + critic as one unified model.
            """
            # Import the same model architecture used in training
            import sys
            sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))))
            from run_all import STSIEEncoder as TrainEncoder, Policy as TrainPolicy, MLPEncoder

            # Reconstruct model with same architecture as training
            # obs_dim matches Env6D: 5*2 + 1 + 1 + 1 + 6 + 6 + 2 = 25
            obs_dim = 5 * meas_dim + 1 + 1 + 1 + state_dim + state_dim + meas_dim
            act_dim = state_dim + meas_dim  # 8 for 6D state + 2D meas
            ctx_len = 30  # innovation buffer length

            enc = TrainEncoder(meas_dim, 4, 32, 16, 4)
            self.full_policy = TrainPolicy(obs_dim, act_dim, enc).eval()

            try:
                ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
                self.full_policy.load_state_dict(ckpt)
                self.get_logger().info(f"Loaded checkpoint: {ckpt_path}")
            except Exception as e:
                self.get_logger().warn(f"Checkpoint load failed: {e}. Using random weights.")
            except FileNotFoundError:
                self.get_logger().warn(
                    f"Checkpoint not found: {ckpt_path}. Using random weights."
                )

        def imu_callback(self, msg: Imu):
            u = np.array([
                msg.linear_acceleration.x,
                msg.linear_acceleration.y,
                msg.angular_velocity.z,
            ])
            self.ekf.predict(u)

        def odom_callback(self, msg: Odometry):
            z = np.array([
                msg.pose.pose.position.x,
                msg.pose.pose.position.y,
            ])
            self.ekf.update(z)
            self._step_count += 1

            # Two-timescale: update encoder every K steps
            if self._step_count % self.K == 0:
                self._update_context()

            # Fast loop: policy with cached z
            self._adapt_noise()
            self._publish_state(msg.header)
            self._publish_diagnostics()

        def _update_context(self):
            """Slow loop: recompute context (handled internally by full_policy encoder)."""
            # With the unified policy from run_all.py, the encoder runs as part of forward()
            # No separate context caching needed — just track innovation buffer
            pass

        def _adapt_noise(self):
            """Fast loop: run trained policy to adapt Q/R."""
            # Build observation matching training format (from run_all.py Env6D._obs())
            innov_window = self.ekf.get_innovation_window(self.innov_window_size)
            n_innov = 5
            recent_innovs = innov_window[-n_innov:] if len(innov_window) >= n_innov else innov_window

            # Observation: [last 5 innovations flat, nees/6, nis/2, log(tr(P)), log(diag(P)), log(diag(Q)), log(diag(R))]
            iv = np.array(recent_innovs).flatten()
            if len(iv) < n_innov * self.ekf.m:
                iv = np.pad(iv, (0, n_innov * self.ekf.m - len(iv)))

            obs = np.clip(np.concatenate([
                iv,
                [self.ekf.state.nees / 6.0, self.ekf.state.nis / 2.0,
                 np.log1p(max(np.trace(self.ekf.state.P), 0))],
                np.log1p(np.maximum(np.diag(self.ekf.state.P), 0)),
                np.log1p(np.maximum(np.diag(self.ekf.state.Q), 0)),
                np.log1p(np.maximum(np.diag(self.ekf.state.R), 0)),
            ]), -20, 20).astype(np.float32)

            # Innovation buffer for encoder
            ib = np.array(innov_window[-30:] if len(innov_window) >= 30
                          else [[0.0]*self.ekf.m]*30, dtype=np.float32)

            # Filter state for encoder
            fs = np.array([
                self.ekf.state.nees / 6.0, self.ekf.state.nis / 2.0,
                np.log1p(np.trace(self.ekf.state.P)),
                np.log1p(self.ekf.state.S[0, 0] if self.ekf.state.S.shape[0] > 0 else 1.0),
            ], dtype=np.float32)

            with torch.no_grad():
                ot = torch.tensor(obs).unsqueeze(0)
                ibt = torch.tensor(ib).unsqueeze(0)
                fst = torch.tensor(fs).unsqueeze(0)
                dist, _, _ = self.full_policy(ot, ibt, fst)
                action = dist.mean.squeeze(0).numpy()

            # Apply action (clip to [-2, 2] matching training)
            action = np.clip(action, -2.0, 2.0)
            alphas = np.exp(action)

            n_q = self.ekf.n
            Q_scales = alphas[:n_q]
            R_scales = alphas[n_q:]

            Q_nom = np.eye(n_q) * 0.05  # nominal from training
            R_nom = np.eye(self.ekf.m) * 0.5

            Q_new = np.diag(Q_scales) @ Q_nom
            R_new = np.diag(R_scales) @ R_nom
            self.ekf.set_noise(Q_new, R_new)

        def _publish_state(self, header):
            msg = PoseWithCovarianceStamped()
            msg.header = header
            msg.header.frame_id = "odom"
            msg.pose.pose.position.x = float(self.ekf.state.x[0])
            msg.pose.pose.position.y = float(self.ekf.state.x[1])
            cov_flat = self.ekf.state.P.flatten().tolist()
            msg.pose.covariance = cov_flat + [0.0] * (36 - len(cov_flat))
            self.state_pub.publish(msg)

        def _publish_diagnostics(self):
            msg = Float64MultiArray()
            msg.data = [
                self.ekf.state.nees,
                self.ekf.state.nis,
                float(np.trace(self.ekf.state.P)),
                self.ekf.state.Q.diagonal().tolist()[0],
                self.ekf.state.R.diagonal().tolist()[0],
            ]
            self.diagnostics_pub.publish(msg)


def main(args=None):
    if not ROS2_AVAILABLE:
        print("ROS2 and PyTorch required.")
        return
    rclpy.init(args=args)
    node = MetaEKFNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
 