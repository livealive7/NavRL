"""Republishes a depth Image with its header.stamp set to this node's clock.

Isaac Sim's Render/Replicator camera writers and its vehicle-state
publishers stamp messages from two unrelated clocks, off by roughly 1.8e9
seconds. NavRL's map_manager and dynamic detector pair the depth image with
the odometry through message_filters::ApproximateTime, which can never match
across a gap that large. Re-stamping both sides with each bridge node's own
now() (see odom_bridge_node) puts them back in one clock domain, at the cost
of the small delay between Isaac publishing the frame and this node relaying
it.
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import (QoSDurabilityPolicy, QoSHistoryPolicy, QoSProfile,
                       QoSReliabilityPolicy)
from sensor_msgs.msg import Image


class DepthRestampNode(Node):

    def __init__(self):
        super().__init__('depth_restamp_node')

        self.declare_parameter('input_topic', 'camera/depth')
        self.declare_parameter('output_topic', 'camera/depth_restamped')

        input_topic = self.get_parameter('input_topic').value
        output_topic = self.get_parameter('output_topic').value

        # Isaac's depth writer publishes RELIABLE/VOLATILE, and NavRL's
        # message_filters subscribers default to RELIABLE as well.
        depth_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.VOLATILE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10,
        )

        self.pub = self.create_publisher(Image, output_topic, depth_qos)
        self.sub = self.create_subscription(
            Image, input_topic, self.callback, depth_qos)

        self.get_logger().info(
            f'depth_restamp_node: {input_topic} -> {output_topic}')

    def callback(self, msg: Image):
        msg.header.stamp = self.get_clock().now().to_msg()
        self.pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = DepthRestampNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
