"""
The evaluation ROS2 node.
This calculates and publishes real-time NEES, NIS, RMSE metrics
by comparing EKF output against ground truth.
"""

import numpy as np

try:
    import rclpy
    from rclpy.node import Node
    from geometry_msgs.msg import PoseWithCovarianceStamped
    from nav_msgs.msg import Odometry
    from std_msgs.msg import Float64MultiArray
    ROS2_AVAILABLE = True
except ImportError:
    ROS2_AVAILABLE = False

from meta_rl.utils.metrics import compute_nees, chi2_bounds


if ROS2_AVAILABLE:
    class EvaluationNode(Node):
        """Real-time evaluation node comparing EKF output to ground truth."""

        def __init__(self):
            super().__init__("evaluation_node")

            self.declare_parameter("state_dim", 6)
            self.declare_parameter("ekf_topic", "/meta_ekf/state")
            self.declare_parameter("gt_topic", "/ground_truth/odom")

            state_dim = self.get_parameter("state_dim").value
            ekf_topic = self.get_parameter("ekf_topic").value
            gt_topic = self.get_parameter("gt_topic").value

            self.state_dim = state_dim
            self.chi2_lb, self.chi2_ub = chi2_bounds(state_dim)

            self._latest_ekf = None
            self._latest_gt = None
            self._nees_history = []

            self.create_subscription(
                PoseWithCovarianceStamped, ekf_topic, self._ekf_cb, 10
            )
            self.create_subscription(
                Odometry, gt_topic, self._gt_cb, 10
            )

            self.metrics_pub = self.create_publisher(
                Float64MultiArray, "/evaluation/metrics", 10
            )

            # Periodic evaluation
            self.create_timer(0.1, self._evaluate)

            self.get_logger().info("Evaluation node initialized")

        def _ekf_cb(self, msg):
            self._latest_ekf = msg

        def _gt_cb(self, msg):
            self._latest_gt = msg

        def _evaluate(self):
            if self._latest_ekf is None or self._latest_gt is None:
                return

            # Extract EKF state and covariance
            ekf_pos = self._latest_ekf.pose.pose
            x_est = np.array([ekf_pos.position.x, ekf_pos.position.y])

            cov_flat = np.array(self._latest_ekf.pose.covariance)
            P = cov_flat[:4].reshape(2, 2)  # 2x2 position covariance

            # Extract ground truth
            gt_pos = self._latest_gt.pose.pose.position
            x_true = np.array([gt_pos.x, gt_pos.y])

            # Compute NEES (position only)
            nees = compute_nees(x_true, x_est, P)
            self._nees_history.append(nees)

            consistent = self.chi2_lb <= nees <= self.chi2_ub
            rmse = float(np.linalg.norm(x_true - x_est))

            # Running consistency rate
            n = len(self._nees_history)
            nees_arr = np.array(self._nees_history)
            consistency_rate = float(np.mean(
                (nees_arr >= self.chi2_lb) & (nees_arr <= self.chi2_ub)
            ))

            # Publish metrics
            msg = Float64MultiArray()
            msg.data = [nees, rmse, float(consistent), consistency_rate]
            self.metrics_pub.publish(msg)


def main(args=None):
    if not ROS2_AVAILABLE:
        print("ROS2 not available.")
        return
    rclpy.init(args=args)
    node = EvaluationNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
