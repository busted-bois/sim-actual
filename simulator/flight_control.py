"""Position -> attitude-rate controller (ported from STUDYONLY flight_control.py)."""

from __future__ import annotations

import math
import time
from collections import namedtuple
from dataclasses import dataclass

AttitudeCommand = namedtuple(
    "AttitudeCommand",
    ["roll_rate_rads", "pitch_rate_rads", "yaw_rate_rads", "thrust"],
)

ATTITUDE_KP = 1.2
RATE_SIGN_ROLL = 1.0
RATE_SIGN_PITCH = -1.0
RATE_SIGN_YAW = -1.0

MAX_TILT_DEG = 20.0
MAX_SPEED_MS = 20.0
KP_TILT = 1.0
HOVER_THRUST = 0.5
KP_ALT = 1.0
MAX_CLIMB_MS = 4.0
MAX_DESCENT_MS = 3.0
KP_VZ = 0.08
KI_VZ = 0.04
MAX_VZ_INTEGRAL = 0.20


@dataclass
class DroneTelemetry:
    pos_x_m: float
    pos_y_m: float
    pos_z_m: float
    vel_x_ms: float
    vel_y_ms: float
    vel_z_ms: float
    roll_rad: float
    pitch_rad: float
    yaw_rad: float


def telemetry_from_data(data: dict) -> DroneTelemetry | None:
    odom = data.get("odometry")
    attitude = data.get("attitude") or {}
    if odom:
        return DroneTelemetry(
            pos_x_m=float(odom["x"]),
            pos_y_m=float(odom["y"]),
            pos_z_m=float(odom.get("z", 0.0)),
            vel_x_ms=float(odom.get("vx", 0.0)),
            vel_y_ms=float(odom.get("vy", 0.0)),
            vel_z_ms=float(odom.get("vz", 0.0)),
            roll_rad=float(attitude.get("roll", 0.0)),
            pitch_rad=float(attitude.get("pitch", 0.0)),
            yaw_rad=float(attitude.get("yaw", data.get("yaw_rad", 0.0))),
        )
    if data.get("pos_ned"):
        pos = data["pos_ned"]
        vel = data.get("vel_ned", (0.0, 0.0, 0.0))
        return DroneTelemetry(
            pos_x_m=float(pos[0]),
            pos_y_m=float(pos[1]),
            pos_z_m=float(pos[2]),
            vel_x_ms=float(vel[0]),
            vel_y_ms=float(vel[1]),
            vel_z_ms=float(vel[2]),
            roll_rad=float(attitude.get("roll", 0.0)),
            pitch_rad=float(attitude.get("pitch", 0.0)),
            yaw_rad=float(attitude.get("yaw", data.get("yaw_rad", 0.0))),
        )
    return None


def _attitude_p_controller(
    target_roll_rad: float,
    target_pitch_rad: float,
    target_yaw_rad: float,
    current_roll_rad: float,
    current_pitch_rad: float,
    current_yaw_rad: float,
) -> tuple[float, float, float]:
    roll_err = target_roll_rad - current_roll_rad
    pitch_err = target_pitch_rad - current_pitch_rad
    yaw_err = (target_yaw_rad - current_yaw_rad + math.pi) % (2 * math.pi) - math.pi
    return (
        RATE_SIGN_ROLL * ATTITUDE_KP * roll_err,
        RATE_SIGN_PITCH * ATTITUDE_KP * pitch_err,
        RATE_SIGN_YAW * ATTITUDE_KP * yaw_err,
    )


class FlightController:
    """Velocity-error tilt + altitude cascade -> attitude rates + thrust."""

    def __init__(self) -> None:
        self._vz_integral = 0.0
        self._vert_last_t_s: float | None = None

    def reset(self) -> None:
        self._vz_integral = 0.0
        self._vert_last_t_s = None

    def compute_attitude_command(
        self,
        telem: DroneTelemetry,
        target_ned_m: tuple[float, float, float],
    ) -> AttitudeCommand:
        pos_x_m, pos_y_m, pos_z_m = telem.pos_x_m, telem.pos_y_m, telem.pos_z_m
        vel_x_ms, vel_y_ms, vel_z_ms = telem.vel_x_ms, telem.vel_y_ms, telem.vel_z_ms
        roll_rad, pitch_rad, yaw_rad = telem.roll_rad, telem.pitch_rad, telem.yaw_rad
        tgt_x_m, tgt_y_m, tgt_z_m = target_ned_m

        north_err_m = tgt_x_m - pos_x_m
        east_err_m = tgt_y_m - pos_y_m
        horiz_dist_m = math.hypot(north_err_m, east_err_m)

        if horiz_dist_m > 0.001:
            speed_scale = min(horiz_dist_m, MAX_SPEED_MS) / horiz_dist_m
            desired_north_ms = north_err_m * speed_scale
            desired_east_ms = east_err_m * speed_scale
        else:
            desired_north_ms = desired_east_ms = 0.0

        north_vel_err_ms = desired_north_ms - vel_x_ms
        east_vel_err_ms = desired_east_ms - vel_y_ms
        target_yaw_rad = yaw_rad

        yaw_std = -yaw_rad
        cos_y, sin_y = math.cos(yaw_std), math.sin(yaw_std)
        body_fwd_err_ms = cos_y * north_vel_err_ms + sin_y * east_vel_err_ms
        body_right_err_ms = -sin_y * north_vel_err_ms + cos_y * east_vel_err_ms

        max_tilt_rad = math.radians(MAX_TILT_DEG)
        target_pitch_rad = max(
            -max_tilt_rad, min(max_tilt_rad, body_fwd_err_ms * KP_TILT)
        )
        target_roll_rad = max(
            -max_tilt_rad, min(max_tilt_rad, body_right_err_ms * KP_TILT)
        )

        vert_err_m = -(tgt_z_m - pos_z_m)
        climb_ms = -vel_z_ms

        now_s = time.time()
        dt_s = 0.0 if self._vert_last_t_s is None else (now_s - self._vert_last_t_s)
        self._vert_last_t_s = now_s

        desired_climb_ms = vert_err_m * KP_ALT
        desired_climb_ms = max(-MAX_DESCENT_MS, min(MAX_CLIMB_MS, desired_climb_ms))

        climb_err_ms = desired_climb_ms - climb_ms
        if KI_VZ > 0:
            self._vz_integral += climb_err_ms * dt_s
            self._vz_integral = max(
                -MAX_VZ_INTEGRAL / KI_VZ,
                min(MAX_VZ_INTEGRAL / KI_VZ, self._vz_integral),
            )
        else:
            self._vz_integral = 0.0

        thrust_correction = climb_err_ms * KP_VZ + self._vz_integral * KI_VZ
        cos_tilt = max(math.cos(target_pitch_rad) * math.cos(target_roll_rad), 0.1)
        thrust = max(0.0, min(1.0, (HOVER_THRUST + thrust_correction) / cos_tilt))

        roll_rate_rads, pitch_rate_rads, yaw_rate_rads = _attitude_p_controller(
            target_roll_rad,
            target_pitch_rad,
            target_yaw_rad,
            roll_rad,
            pitch_rad,
            yaw_rad,
        )
        return AttitudeCommand(roll_rate_rads, pitch_rate_rads, 0.0, thrust)
