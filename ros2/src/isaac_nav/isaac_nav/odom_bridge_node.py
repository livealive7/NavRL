"""Synthesizes nav_msgs/Odometry from Isaac Sim's separate pose and twist topics.

NavRL's navigation node subscribes to a single nav_msgs/Odometry topic and
treats ``twist.twist.linear`` as a BODY-frame velocity (it rotates it into
the world frame itself). Pegasus' ROS2 backend publishes:

  * ``state/pose``   PoseStamped  ENU world frame (BEST_EFFORT)
  * ``state/twist``  TwistStamped FLU body frame  (BEST_EFFORT)
  * ``state/twist_inertial`` TwistStamped ENU world frame

so this node pairs ``state/pose`` with the body-frame ``state/twist`` by
default. Using ``state/twist_inertial`` instead would make NavRL rotate an
already-world-frame velocity a second time and corrupt its velocity
observation whenever the yaw is non-zero.

The output header is re-stamped with this node's clock. Isaac's camera
writers and vehicle-state publishers stamp from unrelated clocks, so
depth_restamp_node does the same on the depth side; both outputs then share
one clock domain and message_filters::ApproximateTime in NavRL's map_manager
can pair them.
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import (QoSDurabilityPolicy, QoSHistoryPolicy, QoSProfile,
                       QoSReliabilityPolicy)

import message_filters
from geometry_msgs.msg import PoseStamped, TwistStamped
from nav_msgs.msg import Odometry


class OdomBridgeNode(Node):

    def __init__(self):
        super().__init__('isaac_odom_bridge')

        self.declare_parameter('pose_topic', 'state/pose')
        self.declare_parameter('twist_topic', 'state/twist')
        self.declare_parameter('output_topic', 'odom')
        # Isaac's own tf tree is rooted at 'map', and every NavRL
        # visualization marker uses 'map' too, so no extra world->map
        # static transform is needed.
        self.declare_parameter('world_frame_id', 'map')
        self.declare_parameter('child_frame_id', '')
        self.declare_parameter('sync_queue_size', 10)
        self.declare_parameter('sync_slop', 0.02)

        pose_topic = self.get_parameter('pose_topic').value
        twist_topic = self.get_parameter('twist_topic').value
        output_topic = self.get_parameter('output_topic').value
        self.world_frame_id = self.get_parameter('world_frame_id').value
        self.child_frame_id = self.get_parameter('child_frame_id').value
        queue_size = self.get_parameter('sync_queue_size').value
        slop = self.get_parameter('sync_slop').value

        # Isaac Sim's ROS2 bridge publishes these as BEST_EFFORT/VOLATILE; a
        # default RELIABLE subscription silently receives nothing.
        sensor_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.VOLATILE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=queue_size,
        )

        self.pose_sub = message_filters.Subscriber(
            self, PoseStamped, pose_topic, qos_profile=sensor_qos)
        self.twist_sub = message_filters.Subscriber(
            self, TwistStamped, twist_topic, qos_profile=sensor_qos)

        self.sync = message_filters.ApproximateTimeSynchronizer(
            [self.pose_sub, self.twist_sub], queue_size, slop)
        self.sync.registerCallback(self.sync_callback)

        # NavRL subscribes with the default (RELIABLE) QoS.
        self.odom_pub = self.create_publisher(Odometry, output_topic, 10)

        self.get_logger().info(
            f'isaac_odom_bridge: {pose_topic} + {twist_topic} -> '
            f'{output_topic} (slop={slop}s)')

    def sync_callback(self, pose_msg: PoseStamped, twist_msg: TwistStamped):
        odom = Odometry()
        odom.header.stamp = self.get_clock().now().to_msg()
        odom.header.frame_id = self.world_frame_id or pose_msg.header.frame_id
        odom.child_frame_id = self.child_frame_id or twist_msg.header.frame_id
        odom.pose.pose = pose_msg.pose
        odom.twist.twist = twist_msg.twist
        self.odom_pub.publish(odom)


def main(args=None):
    rclpy.init(args=args)
    node = OdomBridgeNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
