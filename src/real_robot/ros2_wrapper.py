"""
ROS2 Wrapper for Real Robot Deployment
"""

try:
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import Image, CameraInfo
    from geometry_msgs.msg import Twist
    from nav_msgs.msg import Odometry
    from cv_bridge import CvBridge
    ROS2_AVAILABLE = True
except ImportError:
    ROS2_AVAILABLE = False
    print("Warning: ROS2 not available. Real robot deployment disabled.")

import torch
import numpy as np
from pathlib import Path


class SlingshotNavigatorNode(Node if ROS2_AVAILABLE else object):
    """ROS2 node for slingshot navigation."""

    def __init__(self, checkpoint_path: str, config_path: str, instruction: str):
        if not ROS2_AVAILABLE:
            raise RuntimeError("ROS2 not available")

        super().__init__('slingshot_navigator')

        self.instruction = instruction
        self.bridge = CvBridge()

        # Load model
        from ..slingshot_policy import SlingshotPolicy
        self.policy = SlingshotPolicy()
        checkpoint = torch.load(checkpoint_path)
        self.policy.load_state_dict(checkpoint['model_state_dict'])
        self.policy.eval()

        # Subscribers
        self.rgb_sub = self.create_subscription(
            Image, '/camera/color/image_raw', self.rgb_callback, 10)
        self.depth_sub = self.create_subscription(
            Image, '/camera/depth/image_raw', self.depth_callback, 10)

        # Publisher
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        # State
        self.latest_rgb = None
        self.latest_depth = None

        # Control loop
        self.timer = self.create_timer(0.5, self.control_loop)

        self.get_logger().info(f'Slingshot Navigator started: "{instruction}"')

    def rgb_callback(self, msg):
        self.latest_rgb = self.bridge.imgmsg_to_cv2(msg, "rgb8")

    def depth_callback(self, msg):
        self.latest_depth = self.bridge.imgmsg_to_cv2(msg, "32FC1")

    def control_loop(self):
        if self.latest_rgb is None or self.latest_depth is None:
            return

        # Prepare inputs
        rgb = torch.from_numpy(self.latest_rgb).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        depth = torch.from_numpy(self.latest_depth).unsqueeze(0).unsqueeze(0).float()

        # Run policy
        with torch.no_grad():
            outputs = self.policy(rgb, depth, self.instruction)

        action = outputs["action"]

        # Publish command
        cmd = Twist()
        cmd.linear.x = float(action.get("linear_vel", 0.0))
        cmd.angular.z = float(action.get("angular_vel", 0.0))
        self.cmd_vel_pub.publish(cmd)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--instruction", required=True)
    args = parser.parse_args()

    if ROS2_AVAILABLE:
        rclpy.init()
        node = SlingshotNavigatorNode(args.checkpoint, args.config, args.instruction)
        rclpy.spin(node)
        rclpy.shutdown()
    else:
        print("ROS2 not available. Cannot run real robot node.")
