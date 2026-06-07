import time

from pymavlink import mavutil

from simulator.manual_control import ManualControl
from simulator.pilot import Pilot

MAVLINK_CMD_SIM_RESET = 31000

RATES_ATTITUDE_MASK = mavutil.mavlink.ATTITUDE_TARGET_TYPEMASK_ATTITUDE_IGNORE

VELOCITY_POSITION_MASK = (
    mavutil.mavlink.POSITION_TARGET_TYPEMASK_X_IGNORE
    | mavutil.mavlink.POSITION_TARGET_TYPEMASK_Y_IGNORE
    | mavutil.mavlink.POSITION_TARGET_TYPEMASK_Z_IGNORE
    | mavutil.mavlink.POSITION_TARGET_TYPEMASK_AX_IGNORE
    | mavutil.mavlink.POSITION_TARGET_TYPEMASK_AY_IGNORE
    | mavutil.mavlink.POSITION_TARGET_TYPEMASK_AZ_IGNORE
    | mavutil.mavlink.POSITION_TARGET_TYPEMASK_YAW_IGNORE
)

CONTROL_HZ = 250

VALID_CONTROL_MODES = ("motor", "attitude", "position")


def _send_motor_control(mavlink_conn, motor_rpms):
    mavlink_conn.mav.set_actuator_control_target_send(
        int(time.time() * 1e6),
        mavlink_conn.target_system,
        mavlink_conn.target_component,
        0,
        motor_rpms,
    )


def _send_attitude_rates(
    mavlink_conn, system_boot_ms, roll_rate, pitch_rate, yaw_rate, thrust
):
    now_ms = int(time.time() * 1000)
    mavlink_conn.mav.set_attitude_target_send(
        now_ms - system_boot_ms,
        mavlink_conn.target_system,
        mavlink_conn.target_component,
        RATES_ATTITUDE_MASK,
        [1, 0, 0, 0],
        roll_rate,
        pitch_rate,
        yaw_rate,
        thrust,
    )


def _send_velocity_ned(mavlink_conn, system_boot_ms, vx, vy, vz, yaw_rate=0.0):
    now_ms = int(time.time() * 1000)
    mavlink_conn.mav.set_position_target_local_ned_send(
        now_ms - system_boot_ms,
        mavlink_conn.target_system,
        mavlink_conn.target_component,
        mavutil.mavlink.MAV_FRAME_LOCAL_NED,
        VELOCITY_POSITION_MASK,
        0.0,
        0.0,
        0.0,
        vx,
        vy,
        vz,
        0.0,
        0.0,
        0.0,
        0.0,
        yaw_rate,
    )


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
        self._vx = 2.0
        self._vy = 0.0
        self._vz = 0.0
        self._vel_yaw_rate = 0.0
        self.pilot = Pilot(self, data)
        self.manual = ManualControl(self, data)

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

    def set_velocity_ned(self, vx=2.0, vy=0.0, vz=0.0, yaw_rate=0.0):
        self._vx = vx
        self._vy = vy
        self._vz = vz
        self._vel_yaw_rate = yaw_rate

    def update(self):
        if not self.manual.tick():
            self.pilot.tick()
        if self.control_mode == "motor":
            _send_motor_control(self.sim_conn, self._motor_rpms)
        elif self.control_mode == "attitude":
            _send_attitude_rates(
                self.sim_conn,
                self.system_boot_ms,
                self._roll_rate,
                self._pitch_rate,
                self._yaw_rate,
                self._thrust,
            )
        elif self.control_mode == "position":
            _send_velocity_ned(
                self.sim_conn,
                self.system_boot_ms,
                self._vx,
                self._vy,
                self._vz,
                self._vel_yaw_rate,
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
