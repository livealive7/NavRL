"""Forwards geometry_msgs/Twist body-frame velocity commands to PX4 SITL
over MAVLink using SET_POSITION_TARGET_LOCAL_NED (velocity + yaw-rate only,
MAV_FRAME_BODY_NED).

Twist is in ROS's FLU body convention (X-forward, Y-left, Z-up, positive
yaw-rate = CCW from above). MAV_FRAME_BODY_NED is FRD (X-forward, Y-right,
Z-down, positive yaw-rate = CW from above), so linear.y, linear.z and
angular.z all need a sign flip.

PX4 offboard requirements handled here:
  * Setpoints must be streamed continuously (> 2 Hz) BEFORE switching to
    OFFBOARD, and must keep arriving afterwards, otherwise PX4 drops out of
    OFFBOARD (COM_OF_LOSS_T). A timer therefore publishes at a fixed rate
    and falls back to zero velocity when cmd_vel goes stale.
  * The vehicle has to be armed and put into OFFBOARD mode explicitly.

After arming, the bridge can also climb to a fixed height above the arming
point (``takeoff_height``) before it hands control over to cmd_vel. During the
climb cmd_vel is ignored. Without it, a zero-velocity OFFBOARD setpoint never
leaves the ground and PX4 auto-disarms (COM_DISARM_PRFLT).
"""

import time

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from pymavlink import mavutil

# PX4 custom main mode for OFFBOARD (see PX4 px4_custom_mode.h).
PX4_CUSTOM_MAIN_MODE_OFFBOARD = 6

# SET_POSITION_TARGET_LOCAL_NED type_mask bits.
POSITION_TARGET_TYPEMASK_X_IGNORE = 1
POSITION_TARGET_TYPEMASK_Y_IGNORE = 2
POSITION_TARGET_TYPEMASK_Z_IGNORE = 4
POSITION_TARGET_TYPEMASK_AX_IGNORE = 64
POSITION_TARGET_TYPEMASK_AY_IGNORE = 128
POSITION_TARGET_TYPEMASK_AZ_IGNORE = 256
POSITION_TARGET_TYPEMASK_YAW_IGNORE = 1024

# Enable only vx, vy, vz and yaw_rate.
VELOCITY_YAWRATE_TYPE_MASK = (
    POSITION_TARGET_TYPEMASK_X_IGNORE
    | POSITION_TARGET_TYPEMASK_Y_IGNORE
    | POSITION_TARGET_TYPEMASK_Z_IGNORE
    | POSITION_TARGET_TYPEMASK_AX_IGNORE
    | POSITION_TARGET_TYPEMASK_AY_IGNORE
    | POSITION_TARGET_TYPEMASK_AZ_IGNORE
    | POSITION_TARGET_TYPEMASK_YAW_IGNORE
)

FRAMES = {
    'BODY_NED': mavutil.mavlink.MAV_FRAME_BODY_NED,
    'BODY_OFFSET_NED': mavutil.mavlink.MAV_FRAME_BODY_OFFSET_NED,
}


class CmdVelPx4Bridge(Node):

    def __init__(self):
        super().__init__('cmd_vel_px4_bridge')

        # PX4 SITL sends offboard/onboard MAVLink to UDP 14540 (GCS is 14550).
        self.declare_parameter('connection_string', 'udpin:0.0.0.0:14540')
        self.declare_parameter('cmd_vel_topic', 'cmd_vel')
        self.declare_parameter('setpoint_rate_hz', 20.0)
        self.declare_parameter('cmd_timeout_s', 0.5)
        self.declare_parameter('frame', 'BODY_NED')
        # Automatically arm and switch to OFFBOARD. Intended for simulation;
        # set to False on real hardware and do it from the RC / GCS instead.
        self.declare_parameter('auto_arm_offboard', True)
        # Climb to this height above the point where the vehicle was armed
        # before cmd_vel is forwarded. 0 disables the automatic takeoff.
        self.declare_parameter('takeoff_height', 2.0)
        self.declare_parameter('takeoff_speed', 1.0)
        self.declare_parameter('takeoff_tolerance', 0.15)
        # Odometry (ROS ENU, z up) used to measure the climb.
        self.declare_parameter('odom_topic', 'odom')

        connection_string = self.get_parameter('connection_string').value
        cmd_vel_topic = self.get_parameter('cmd_vel_topic').value
        rate_hz = float(self.get_parameter('setpoint_rate_hz').value)
        self.cmd_timeout = float(self.get_parameter('cmd_timeout_s').value)
        frame_name = self.get_parameter('frame').value
        self.auto_arm_offboard = bool(
            self.get_parameter('auto_arm_offboard').value)
        self.takeoff_height = float(self.get_parameter('takeoff_height').value)
        self.takeoff_speed = float(self.get_parameter('takeoff_speed').value)
        self.takeoff_tol = float(self.get_parameter('takeoff_tolerance').value)
        odom_topic = self.get_parameter('odom_topic').value

        if frame_name not in FRAMES:
            raise ValueError(
                f'Unknown frame "{frame_name}", expected one of {list(FRAMES)}')
        self.frame = FRAMES[frame_name]

        self.get_logger().info(f'Connecting to {connection_string} ...')
        self.master = mavutil.mavlink_connection(connection_string)

        # With udpin: we bind and PX4 sends heartbeats to us on its own, so no
        # registration heartbeat is needed. Only accept the autopilot
        # component, since other components can also heartbeat on this port.
        hb = None
        deadline = time.monotonic() + 30.0
        while time.monotonic() < deadline:
            msg = self.master.recv_match(
                type='HEARTBEAT', blocking=True, timeout=1)
            if msg is None:
                continue
            if (msg.get_srcComponent()
                    == mavutil.mavlink.MAV_COMP_ID_AUTOPILOT1
                    and msg.type != mavutil.mavlink.MAV_TYPE_GCS):
                hb = msg
                break
        if hb is None:
            raise RuntimeError(
                f'No autopilot heartbeat on {connection_string} within 30 s')

        # recv_match() does not populate target ids, set them explicitly.
        self.master.target_system = hb.get_srcSystem()
        self.master.target_component = hb.get_srcComponent()
        self.get_logger().info(
            f'Heartbeat received (system {self.master.target_system} '
            f'component {self.master.target_component})')

        self.armed = False
        self.offboard = False
        self.setpoints_sent = 0
        self.last_mode_request = 0.0

        self.last_cmd = Twist()
        self.last_cmd_time = 0.0

        # Takeoff state: z_ref is the height while still disarmed on the ground.
        self.odom_z = None
        self.z_ref = None
        self.takeoff_done = self.takeoff_height <= 0.0

        self.sub = self.create_subscription(Twist, cmd_vel_topic, self.cb, 10)
        self.odom_sub = self.create_subscription(
            Odometry, odom_topic, self.odom_cb, 10)
        self.timer = self.create_timer(1.0 / rate_hz, self.tick)

    # ------------------------------------------------------------------ ROS

    def cb(self, msg: Twist):
        # Only store the latest command; the timer does the actual sending.
        self.last_cmd = msg
        self.last_cmd_time = time.monotonic()

    def odom_cb(self, msg: Odometry):
        self.odom_z = msg.pose.pose.position.z

    def takeoff_cmd(self):
        """Velocity command for the takeoff climb, or None once it is done."""
        if self.takeoff_done:
            return None
        if not self.armed:
            # Remember the ground height; hold still until armed.
            self.z_ref = self.odom_z
            return Twist()
        if self.odom_z is None or self.z_ref is None:
            return Twist()
        err = self.z_ref + self.takeoff_height - self.odom_z
        if err < self.takeoff_tol:
            self.takeoff_done = True
            self.get_logger().info(
                f'Takeoff done at {self.odom_z - self.z_ref:.2f} m, '
                'forwarding cmd_vel')
            return None
        cmd = Twist()
        # Proportional approach, never below 0.3 m/s so it cannot stall.
        cmd.linear.z = min(self.takeoff_speed, max(0.3, err))
        return cmd

    def tick(self):
        self.poll_heartbeat()
        self.send_setpoint()
        self.setpoints_sent += 1

        # PX4 needs a stream of setpoints (about 1 s worth here) before it
        # accepts OFFBOARD.
        if self.auto_arm_offboard and self.setpoints_sent > 20:
            self.ensure_armed_and_offboard()

    # -------------------------------------------------------------- MAVLink

    def send_setpoint(self):
        # Fall back to zero velocity if cmd_vel is stale so OFFBOARD stays
        # alive but the vehicle holds still.
        takeoff = self.takeoff_cmd()
        if takeoff is not None:
            msg = takeoff
        elif time.monotonic() - self.last_cmd_time > self.cmd_timeout:
            msg = Twist()
        else:
            msg = self.last_cmd

        self.master.mav.set_position_target_local_ned_send(
            0,
            self.master.target_system,
            self.master.target_component,
            self.frame,
            VELOCITY_YAWRATE_TYPE_MASK,
            0, 0, 0,           # x, y, z (ignored)
            msg.linear.x, -msg.linear.y, -msg.linear.z,  # FLU->FRD: y,z flip
            0, 0, 0,           # accel (ignored)
            0, -msg.angular.z,  # yaw (ignored), yaw_rate (FLU CCW -> FRD CW)
        )

    def poll_heartbeat(self):
        # Drain pending messages and track armed / mode state from HEARTBEAT.
        while True:
            hb = self.master.recv_match(type='HEARTBEAT', blocking=False)
            if hb is None:
                break
            if hb.get_srcSystem() != self.master.target_system:
                continue
            if hb.get_srcComponent() != self.master.target_component:
                continue
            was_armed = self.armed
            self.armed = bool(
                hb.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED)
            if was_armed and not self.armed and self.takeoff_height > 0.0:
                self.get_logger().info('Disarmed, will take off again on re-arm')
                self.takeoff_done = False
            # PX4 packs main mode into bits 16-23 of custom_mode.
            main_mode = (hb.custom_mode >> 16) & 0xFF
            self.offboard = main_mode == PX4_CUSTOM_MAIN_MODE_OFFBOARD

    def ensure_armed_and_offboard(self):
        # Retry every 2 s until the heartbeat confirms both states.
        now = time.monotonic()
        if now - self.last_mode_request < 2.0:
            return
        if self.armed and self.offboard:
            return
        self.last_mode_request = now

        if not self.offboard:
            self.set_offboard_mode()
        if not self.armed:
            self.arm()

    def arm(self):
        self.get_logger().info('Sending arm command')
        self.master.mav.command_long_send(
            self.master.target_system, self.master.target_component,
            mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM, 0,
            1, 0, 0, 0, 0, 0, 0)

    def set_offboard_mode(self):
        self.get_logger().info('Requesting OFFBOARD mode')
        self.master.mav.command_long_send(
            self.master.target_system, self.master.target_component,
            mavutil.mavlink.MAV_CMD_DO_SET_MODE, 0,
            mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
            PX4_CUSTOM_MAIN_MODE_OFFBOARD, 0, 0, 0, 0, 0)


def main(args=None):
    rclpy.init(args=args)
    node = CmdVelPx4Bridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()