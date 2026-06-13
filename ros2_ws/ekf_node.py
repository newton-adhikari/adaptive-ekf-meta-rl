"""
This is a simple:EKF ROS2 Node: 
Standard EKF state estimation node.
Subscribes to IMU + odometry, publishes filtered state.
"""


import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu
from nav_msgs.msg import Odometry
from std_msgs.msg import Float64MultiArray
from geometry_msgs.msg import PoseWithCovarianceStamped

from meta_rl.utils.ekf import EKF

class EKFNode(Node):
    """ROS2 EKF node for state estimation."""

    def __init__(self):
        super().__init__("ekf_node")

        # Parameters
        self.declare_parameter("dt", 0.01)
        self.declare_parameter("state_dim", 6)
        self.declare_parameter("meas_dim", 2)

        dt = self.get_parameter("dt").value
        state_dim = self.get_parameter("state_dim").value
        meas_dim = self.get_parameter("meas_dim").value

        self.ekf = EKF(state_dim, meas_dim, dt)
        self.ekf.reset(
            x0=np.zeros(state_dim),
            P0=np.eye(state_dim) * 0.5,
            Q=np.eye(state_dim) * 0.1,
            R=np.eye(meas_dim) * 1.0,
        )

        self.create_subscription(Imu, "/imu/data", self.imu_callback, 10)
        self.create_subscription(Odometry, "/odom", self.odom_callback, 10)

        # Publishers

        self.state_pub = self.create_publisher(
            PoseWithCovarianceStamped, "/ekf/state", 10
        )
        self.diagnostics_pub = self.create_publisher(
            Float64MultiArray, "/ekf/diagnostics", 10
        )

        self.get_logger().info("EKF node initialized")


    def imu_callback(self, msg: Imu):
        """
        here i use IMU for prediction step.
        """
        u = np.array([
            msg.linear_acceleration.x,
            msg.linear_acceleration.y,
            msg.angular_velocity.z,
        ])
        self.ekf.predict(u)

    def odom_callback(self, msg: Odometry):
        """
        here i use odometry position for update step.
        """
        z = np.array([
            msg.pose.pose.position.x,
            msg.pose.pose.position.y,
        ])
        self.ekf.update(z)
        self._publish_state(msg.header)
        self._publish_diagnostics()

    def _publish_state(self, header):
        msg = PoseWithCovarianceStamped()
        msg.header = header
        msg.header.frame_id = "odom"
        msg.pose.pose.position.x = float(self.ekf.state.x[0])
        msg.pose.pose.position.y = float(self.ekf.state.x[1])
        # Flatten 6x6 covariance to 36-element array (ROS convention)
        cov_flat = self.ekf.state.P.flatten().tolist()
        msg.pose.covariance = cov_flat + [0.0] * (36 - len(cov_flat))
        self.state_pub.publish(msg)

    def _publish_diagnostics(self):
        msg = Float64MultiArray()
        msg.data = [
            self.ekf.state.nees,
            self.ekf.state.nis,
            float(np.trace(self.ekf.state.P)),
        ]
        self.diagnostics_pub.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    
    rclpy.shutdown()


if __name__ == "__main__":
    main()
