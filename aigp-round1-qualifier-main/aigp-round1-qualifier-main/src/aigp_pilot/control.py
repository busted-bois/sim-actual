from __future__ import annotations

import math

import numpy as np

G = 9.81


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def wrap_to_pi(angle: float) -> float:
    return (angle + math.pi) % (2 * math.pi) - math.pi


def accel_to_tilt(a_north: float, a_east: float, yaw: float, *, tilt_max: float) -> tuple[float, float]:
    cos_yaw, sin_yaw = math.cos(yaw), math.sin(yaw)
    body_forward = a_north * cos_yaw + a_east * sin_yaw
    body_right = -a_north * sin_yaw + a_east * cos_yaw
    pitch = clamp(body_forward / G, -tilt_max, tilt_max)
    roll = clamp(body_right / G, -tilt_max, tilt_max)
    return roll, pitch


def vertical_accel(
    target_z: float,
    pos_z: float,
    vel_z: float,
    *,
    vz_ff: float,
    kpz_pos: float,
    kpz_vel: float,
    vz_max: float = 4.5,
    az_max: float = 10.0,
) -> float:
    vz = clamp(vz_ff + kpz_pos * (target_z - pos_z), -vz_max, vz_max)
    return clamp(kpz_vel * (vz - vel_z), -az_max, az_max)


def collective_thrust(
    az: float,
    roll: float,
    pitch: float,
    *,
    hover: float,
    thr_min: float = 0.05,
    thr_max: float = 0.98,
) -> float:
    base = hover * (G - az) / G
    tilt_factor = max(math.cos(roll) * math.cos(pitch), 0.5)
    return clamp(base / tilt_factor, thr_min, thr_max)


def attitude_rate_command(measured_rpy, desired_rpy, *, kp: float, max_rate: float) -> np.ndarray:
    rates = np.array(
        [
            kp * (measured_rpy[0] - desired_rpy[0]),
            kp * (measured_rpy[1] - desired_rpy[1]),
            kp * wrap_to_pi(measured_rpy[2] - desired_rpy[2]),
        ],
        dtype=float,
    )
    return np.clip(rates, -max_rate, max_rate)
