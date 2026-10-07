"""Forwards geometry_msgs/Twist body-frame velocity commands to ArduPilot
SITL over MAVLink using SET_POSITION_TARGET_LOCAL_NED (velocity + yaw-rate
only, MAV_FRAME_BODY_OFFSET_NED).

Twist is in ROS's FLU body convention (X-forward, Y-left, Z-up, positive
yaw-rate = CCW from above). MAV_FRAME_BODY_OFFSET_NED is FRD (X-forward,
Y-right, Z-down, positive yaw-rate = CW from above). Confirmed directly
from Isaac Sim's own published tf: drone_base_link -> drone_base_link_frd
is a 180 deg rotation about X (quaternion x=1,y=0,z=0,w=0), which negates
Y and Z and, applied to angular velocity, negates yaw-rate too -- so all
three of linear.y, linear.z and angular.z need the sign flip, not just z.
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from pymavlink import mavutil


class CmdVelArduPilotBridge(Node):

    def __init__(self):
        super().__init__('cmd_vel_ardupilot_bridge')

        self.declare_parameter('connection_string', 'udp:127.0.0.1:14550')
        self.declare_parameter('cmd_vel_topic', 'cmd_vel')

        connection_string = self.get_parameter('connection_string').value
        cmd_vel_topic = self.get_parameter('cmd_vel_topic').value

        self.get_logger().info(f'Connecting to {connection_string} ...')
        self.master = mavutil.mavlink_connection(connection_string)
        # When connection_string is udpout: (a connect()ed client socket, as
        # opposed to udpin:/bare udp: which bind()/listen()), the peer we're
        # connecting to (MAVProxy's own --out udpin:... endpoint) only learns
        # our address -- and therefore only starts routing anything back to
        # us, including the heartbeats wait_heartbeat() is waiting for --
        # once it has received at least one packet FROM us. wait_heartbeat()
        # itself never sends anything, so on a fresh socket (new ephemeral
        # source port every process start) this can hang indefinitely.
        # Proactively send our own heartbeat first to register.
        for _ in range(50):
            self.master.mav.heartbeat_send(
                mavutil.mavlink.MAV_TYPE_GCS,
                mavutil.mavlink.MAV_AUTOPILOT_INVALID, 0, 0, 0)
            hb = self.master.recv_match(type='HEARTBEAT', blocking=True, timeout=1)
            if hb and hb.get_srcSystem() != self.master.source_system:
                break
        else:
            raise RuntimeError(
                f'No heartbeat from ArduPilot on {connection_string} after '
                'repeated registration attempts')
        # wait_heartbeat()/recv_match() only return the HEARTBEAT message --
        # they do NOT populate target_system/target_component (those are
        # pymavlink properties backed by self.sysid / self.param_sysid, which
        # start at 0 and are never auto-updated from received traffic
        # anywhere in mavutil). Left at the default target_component=0, every
        # message this node sends -- including SET_POSITION_TARGET_LOCAL_NED
        # below -- goes out with the wrong component id and ArduCopter
        # silently drops it, so the vehicle never actually moves despite
        # cmd_vel flowing. Confirmed by direct testing: identical traffic
        # with target_component forced to the heartbeat's source component
        # (1, MAV_COMP_ID_AUTOPILOT1) elicits real velocity response; left at
        # the default 0 it doesn't.
        self.master.target_system = hb.get_srcSystem()
        self.master.target_component = hb.get_srcComponent()
        self.get_logger().info(
            f'Heartbeat received (system {self.master.target_system} '
            f'component {self.master.target_component})')

        self.sub = self.create_subscription(Twist, cmd_vel_topic, self.cb, 10)

    def cb(self, msg: Twist):
        self.get_logger().debug(
            f'linear: x={msg.linear.x:.2f} y={msg.linear.y:.2f} z={msg.linear.z:.2f} '
            f'| angular: z={msg.angular.z:.2f}')
        # type_mask: enable only vx, vy, vz, yaw_rate (ignore position/accel/yaw).
        # Spelled out via named constants rather than a hand-rolled binary
        # literal -- 0b0000_0111_1100_0111 (the previous value here) silently
        # also set POSITION_TARGET_TYPEMASK_FORCE_SET (bit 9), which tells
        # ArduPilot to treat the (unused, zeroed) acceleration fields as a
        # force setpoint instead of ignoring them; that's not something
        # ArduCopter's velocity controller expects here and was preventing
        # the vehicle from actually tracking the commanded velocity.
        POSITION_TARGET_TYPEMASK_X_IGNORE = 1
        POSITION_TARGET_TYPEMASK_Y_IGNORE = 2
        POSITION_TARGET_TYPEMASK_Z_IGNORE = 4
        POSITION_TARGET_TYPEMASK_AX_IGNORE = 64
        POSITION_TARGET_TYPEMASK_AY_IGNORE = 128
        POSITION_TARGET_TYPEMASK_AZ_IGNORE = 256
        POSITION_TARGET_TYPEMASK_YAW_IGNORE = 1024
        type_mask = (
            POSITION_TARGET_TYPEMASK_X_IGNORE
            | POSITION_TARGET_TYPEMASK_Y_IGNORE
            | POSITION_TARGET_TYPEMASK_Z_IGNORE
            | POSITION_TARGET_TYPEMASK_AX_IGNORE
            | POSITION_TARGET_TYPEMASK_AY_IGNORE
            | POSITION_TARGET_TYPEMASK_AZ_IGNORE
            | POSITION_TARGET_TYPEMASK_YAW_IGNORE
        )
        self.master.mav.set_position_target_local_ned_send(
            0,
            self.master.target_system,
            self.master.target_component,
            mavutil.mavlink.MAV_FRAME_BODY_OFFSET_NED,
            type_mask,
            0, 0, 0,           # x, y, z (ignored)
            msg.linear.x, -msg.linear.y, -msg.linear.z,  # FLU->FRD: y,z flip
            0, 0, 0,           # accel (ignored)
            0, -msg.angular.z,  # yaw, yaw_rate (FLU CCW -> FRD CW)
        )


def main(args=None):
    rclpy.init(args=args)
    node = CmdVelArduPilotBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
