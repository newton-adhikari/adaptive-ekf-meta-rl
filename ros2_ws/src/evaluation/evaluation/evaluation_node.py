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
            self.declare_parameter("gt_topic", "/odom")
            self.declare_parameter("output_file", "")
            self.declare_parameter("method_name", "unknown")

            state_dim = self.get_parameter("state_dim").value
            ekf_topic = self.get_parameter("ekf_topic").value
            gt_topic = self.get_parameter("gt_topic").value
            self.output_file = self.get_parameter("output_file").value
            self.method_name = self.get_parameter("method_name").value

            self.state_dim = state_dim
            self.chi2_lb, self.chi2_ub = chi2_bounds(state_dim)

            self._latest_ekf = None
            self._latest_gt = None
            self._nees_history = []
            self._rmse_history = []

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

            # pose.covariance is 6x6 row-major; extract 2x2 position block
            cov_flat = np.array(self._latest_ekf.pose.covariance)
            P = np.array([
                [cov_flat[0], cov_flat[1]],
                [cov_flat[6], cov_flat[7]],
            ])

            # Skip if covariance is degenerate (early startup)
            if np.trace(P) < 1e-10:
                return

            # Extract ground truth
            gt_pos = self._latest_gt.pose.pose.position
            x_true = np.array([gt_pos.x, gt_pos.y])

            # Compute NEES (position only)
            nees = compute_nees(x_true, x_est, P)
            if np.isnan(nees):
                return
            self._nees_history.append(nees)

            rmse = float(np.linalg.norm(x_true - x_est))
            self._rmse_history.append(rmse)

            consistent = self.chi2_lb <= nees <= self.chi2_ub

            # Running consistency rate
            nees_arr = np.array(self._nees_history)
            consistency_rate = float(np.mean(
                (nees_arr >= self.chi2_lb) & (nees_arr <= self.chi2_ub)
            ))

            # Publish metrics
            msg = Float64MultiArray()
            msg.data = [nees, rmse, float(consistent), consistency_rate]
            self.metrics_pub.publish(msg)

        def save_results(self):
            """Save accumulated metrics to JSON file."""
            if not self.output_file or len(self._nees_history) == 0:
                return

            nees_arr = np.array(self._nees_history)
            rmse_arr = np.array(self._rmse_history)

            results = {
                "method": self.method_name,
                "num_samples": len(self._nees_history),
                "mean_nees": float(np.mean(nees_arr)),
                "std_nees": float(np.std(nees_arr)),
                "median_nees": float(np.median(nees_arr)),
                "mean_rmse": float(np.mean(rmse_arr)),
                "std_rmse": float(np.std(rmse_arr)),
                "max_rmse": float(np.max(rmse_arr)),
                "consistency_rate": float(np.mean(
                    (nees_arr >= self.chi2_lb) & (nees_arr <= self.chi2_ub)
                )),
                "chi2_bounds": [float(self.chi2_lb), float(self.chi2_ub)],
                "nees_history": nees_arr.tolist(),
                "rmse_history": rmse_arr.tolist(),
            }

            with open(self.output_file, "w") as f:
                json.dump(results, f, indent=2)

            self.get_logger().info(
                f"Saved results to {self.output_file} "
                f"(NEES={results['mean_nees']:.4f}, "
                f"Consistency={results['consistency_rate']:.2%}, "
                f"RMSE={results['mean_rmse']:.4f})"
            )


def main(args=None):
    if not ROS2_AVAILABLE:
        print("ROS2 not available.")
        return
    rclpy.init(args=args)
    node = EvaluationNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        node.save_results()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
 