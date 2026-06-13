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

class EKFNode(Node):
    """ROS2 EKF node for state estimation."""

    def __init__(self):
        super().__init__("ekf_node")


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

def main(args=None):
    rclpy.init(args=args)
    
    rclpy.shutdown()


if __name__ == "__main__":
    main()
