import math
import time

from pymavlink import mavutil

from simulator.mavlink_masks import (
    build_attitude_only_type_mask,
    build_body_rate_type_mask,
    build_position_target_type_mask,
    build_position_type_mask,
    build_velocity_type_mask,
)
from simulator.pilot import Pilot

MAVLINK_CMD_SIM_RESET = 31000

CONTROL_HZ = 250
DEFAULT_HIGHRES_IMU_HZ = 120

VALID_CONTROL_MODES = (
    "motor",
    "attitude",
    "position",
    "attitude_pose",
    "position_pose",
)


def _mavlink_frame(frame_name):
    if frame_name == "body_ned":
        return mavutil.mavlink.MAV_FRAME_BODY_NED
    return mavutil.mavlink.MAV_FRAME_LOCAL_NED


def _send_motor_control(mavlink_conn, motor_rpms):
    mavlink_conn.mav.set_actuator_control_target_send(
        int(time.time() * 1e6),
        mavlink_conn.target_system,
        mavlink_conn.target_component,
        0,
        motor_rpms,
    )


def _send_attitude_target(
    mavlink_conn,
    system_boot_ms,
    type_mask,
    quaternion,
    roll_rate,
    pitch_rate,
    yaw_rate,
    thrust,
):
    now_ms = int(time.time() * 1000)
    mavlink_conn.mav.set_attitude_target_send(
        now_ms - system_boot_ms,
        mavlink_conn.target_system,
        mavlink_conn.target_component,
        type_mask,
        quaternion,
        roll_rate,
        pitch_rate,
        yaw_rate,
        thrust,
    )


def _send_position_target(
    mavlink_conn,
    system_boot_ms,
    frame,
    type_mask,
    x,
    y,
    z,
    vx,
    vy,
    vz,
    yaw,
    yaw_rate,
):
    now_ms = int(time.time() * 1000)
    mavlink_conn.mav.set_position_target_local_ned_send(
        now_ms - system_boot_ms,
        mavlink_conn.target_system,
        mavlink_conn.target_component,
        frame,
        type_mask,
        x,
        y,
        z,
        vx,
        vy,
        vz,
        0.0,
        0.0,
        0.0,
        yaw,
        yaw_rate,
    )


def _euler_to_quaternion(roll, pitch, yaw):
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    qw = cr * cp * cy + sr * sp * sy
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy
    return [qw, qx, qy, qz]


class Controller:
    def __init__(self, sim_conn, data, system_boot_ms):
        self.sim_conn = sim_conn
        self.data = data
        self.system_boot_ms = system_boot_ms
        self.control_mode = "attitude"
        self._motor_rpms = [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self._roll_rate = 0.0
        self._pitch_rate = -0.3
        self._yaw_rate = 0.0
        self._thrust = 0.6
        self._roll = 0.0
        self._pitch = 0.0
        self._yaw = 0.0
        self._vx = 2.0
        self._vy = 0.0
        self._vz = 0.0
        self._px = 0.0
        self._py = 0.0
        self._pz = 0.0
        self._position_yaw = 0.0
        self._position_frame = "local_ned"
        self._quaternion = [1.0, 0.0, 0.0, 0.0]
        self._use_explicit_quaternion = False
        self.pilot = Pilot(self, data)

    def set_control_mode(self, mode):
        if mode not in VALID_CONTROL_MODES:
            raise ValueError(f"control_mode must be one of {VALID_CONTROL_MODES}")
        self.control_mode = mode

    def set_motor_rpms(self, front_left, front_right, back_left, back_right):
        self._motor_rpms = [
            front_left,
            front_right,
            back_left,
            back_right,
            0.0,
            0.0,
            0.0,
            0.0,
        ]

    def set_attitude_rates(
        self, roll_rate=0.0, pitch_rate=-0.3, yaw_rate=0.0, thrust=0.6
    ):
        self._roll_rate = roll_rate
        self._pitch_rate = pitch_rate
        self._yaw_rate = yaw_rate
        self._thrust = thrust
        self.control_mode = "attitude"

    def set_attitude_pose(self, roll=0.0, pitch=0.0, yaw=0.0, thrust=0.6):
        self._roll = roll
        self._pitch = pitch
        self._yaw = yaw
        self._thrust = thrust
        self.control_mode = "attitude_pose"

    def set_attitude_quaternion(self, qw, qx, qy, qz, thrust=0.6):
        self._roll = 0.0
        self._pitch = 0.0
        self._yaw = 0.0
        self._thrust = thrust
        self._quaternion = [qw, qx, qy, qz]
        self.control_mode = "attitude_pose"
        self._use_explicit_quaternion = True

    def set_velocity_ned(self, vx=2.0, vy=0.0, vz=0.0):
        self._vx = vx
        self._vy = vy
        self._vz = vz
        self.control_mode = "position"

    def set_velocity_body_ned(self, vx=0.0, vy=0.0, vz=0.0):
        self._vx = vx
        self._vy = vy
        self._vz = vz
        self._position_frame = "body_ned"
        self.control_mode = "position"

    def set_position_ned(self, x=0.0, y=0.0, z=0.0, yaw=0.0):
        self._px = x
        self._py = y
        self._pz = z
        self._position_yaw = yaw
        self._position_frame = "local_ned"
        self.control_mode = "position_pose"

    def request_message_interval(self, message_name, hz):
        message_id = getattr(
            mavutil.mavlink, f"MAVLINK_MSG_ID_{message_name.upper()}", None
        )
        if message_id is None:
            raise ValueError(f"Unknown MAVLink message: {message_name}")
        interval_us = int(1e6 / max(1.0, hz))
        self.sim_conn.mav.command_long_send(
            self.sim_conn.target_system,
            self.sim_conn.target_component,
            mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL,
            0,
            message_id,
            interval_us,
            0,
            0,
            0,
            0,
            0,
        )

    def request_highres_imu(self, hz=DEFAULT_HIGHRES_IMU_HZ):
        self.request_message_interval("HIGHRES_IMU", hz)

    def get_tracking_snapshot(self):
        return self.data.get("tracking_snapshot")

    def update(self):
        self.pilot.tick()
        tracker = self.data.get("_local_tracker")
        if tracker is not None:
            tracker.tick(self.data)

        if self.control_mode == "motor":
            _send_motor_control(self.sim_conn, self._motor_rpms)
        elif self.control_mode == "attitude":
            _send_attitude_target(
                self.sim_conn,
                self.system_boot_ms,
                build_body_rate_type_mask(),
                [1, 0, 0, 0],
                self._roll_rate,
                self._pitch_rate,
                self._yaw_rate,
                self._thrust,
            )
        elif self.control_mode == "attitude_pose":
            if getattr(self, "_use_explicit_quaternion", False):
                quaternion = self._quaternion
            else:
                quaternion = _euler_to_quaternion(self._roll, self._pitch, self._yaw)
            _send_attitude_target(
                self.sim_conn,
                self.system_boot_ms,
                build_attitude_only_type_mask(),
                quaternion,
                0.0,
                0.0,
                0.0,
                self._thrust,
            )
            self._use_explicit_quaternion = False
        elif self.control_mode == "position":
            frame = _mavlink_frame(self._position_frame)
            _send_position_target(
                self.sim_conn,
                self.system_boot_ms,
                frame,
                build_velocity_type_mask(),
                0.0,
                0.0,
                0.0,
                self._vx,
                self._vy,
                self._vz,
                0.0,
                0.0,
            )
            self._position_frame = "local_ned"
        elif self.control_mode == "position_pose":
            frame = _mavlink_frame(self._position_frame)
            type_mask = build_position_type_mask()
            if self._position_yaw != 0.0:
                type_mask = build_position_target_type_mask(
                    use_position=True, use_yaw=True
                )
            _send_position_target(
                self.sim_conn,
                self.system_boot_ms,
                frame,
                type_mask,
                self._px,
                self._py,
                self._pz,
                0.0,
                0.0,
                0.0,
                self._position_yaw,
                0.0,
            )
        time.sleep(1.0 / CONTROL_HZ)

    def arm(self):
        self.sim_conn.mav.command_long_send(
            self.sim_conn.target_system,
            self.sim_conn.target_component,
            mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
            0,
            1,
            0,
            0,
            0,
            0,
            0,
            0,
        )

    def disarm(self):
        self.sim_conn.mav.command_long_send(
            self.sim_conn.target_system,
            self.sim_conn.target_component,
            mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
        )

    def reset_sim(self):
        self.sim_conn.mav.command_long_send(
            self.sim_conn.target_system,
            self.sim_conn.target_component,
            MAVLINK_CMD_SIM_RESET,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
        )
