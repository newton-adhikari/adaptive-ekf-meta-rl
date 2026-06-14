"""
This is the main online CC-MetaEKF ROS2 Node.
which runs the trained meta-RL policy for real-time EKF noise adaptation.

"""

import numpy as np

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
            """Load encoder and policy from checkpoint."""
            latent_dim = 32
            filter_state_dim = 1 + 1 + 1 + meas_dim

            self.encoder = STSIEEncoder(
                innovation_dim=meas_dim,
                filter_state_dim=filter_state_dim,
                latent_dim=latent_dim,
            ).eval()

            obs_dim = self.innov_window_size * meas_dim + 1 + 1 + 1 + meas_dim
            self.policy = GaussianPolicy(
                state_dim=obs_dim,
                context_dim=latent_dim,
                action_dim=state_dim + meas_dim,
            ).eval()

            try:
                ckpt = torch.load(ckpt_path, map_location="cpu")
                self.encoder.load_state_dict(ckpt["encoder"])
                self.policy.load_state_dict(ckpt["policy"])
                self.get_logger().info(f"Loaded checkpoint: {ckpt_path}")
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
            """Slow loop: recompute context z from innovation buffer."""
            innov_window = self.ekf.get_innovation_window(self.innov_window_size)
            filter_state = np.array([
                self.ekf.state.nees, self.ekf.state.nis,
                np.trace(self.ekf.state.P), *np.diag(self.ekf.state.S),
            ])

            with torch.no_grad():
                iw = torch.tensor(innov_window, dtype=torch.float32).unsqueeze(0)
                fs = torch.tensor(filter_state, dtype=torch.float32).unsqueeze(0)
                self._cached_z = self.encoder(iw, fs)

        def _adapt_noise(self):
            """Fast loop: run policy with cached context."""
            innov_window = self.ekf.get_innovation_window(self.innov_window_size)
            filter_state = np.array([
                self.ekf.state.nees, self.ekf.state.nis,
                np.trace(self.ekf.state.P), *np.diag(self.ekf.state.S),
            ])

            obs = np.concatenate([
                innov_window.flatten(),
                [self.ekf.state.nees, self.ekf.state.nis,
                 np.trace(self.ekf.state.P)],
                np.diag(self.ekf.state.S),
            ])

            with torch.no_grad():
                s = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
                action = self.policy.deterministic(s, self._cached_z)
                action = action.numpy().squeeze(0)

            # Apply action
            n_q = self.ekf.n
            Q_scales = np.clip(action[:n_q], 0.1, 10.0)
            R_scales = np.clip(action[n_q:], 0.1, 10.0)

            Q_new = np.diag(Q_scales) @ self.ekf.state.Q
            R_new = np.diag(R_scales) @ self.ekf.state.R
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
