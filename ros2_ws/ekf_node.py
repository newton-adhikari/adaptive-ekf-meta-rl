"""
This is a simple:EKF ROS2 Node: 
Standard EKF state estimation node.
Subscribes to IMU + odometry, publishes filtered state.
"""


import numpy as np
import rclpy
from rclpy.node import Node

class EKFNode(Node):
    """ROS2 EKF node for state estimation."""

    def __init__(self):
        super().__init__("ekf_node")


def main(args=None):
    rclpy.init(args=args)
    
    rclpy.shutdown()


if __name__ == "__main__":
    main()
