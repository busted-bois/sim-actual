import math
import colorsys

import numpy as np


def hex_to_hsv_lower_upper(
    hex_color: str, tolerance: int
) -> tuple[np.ndarray, np.ndarray]:
    hex_color = hex_color.lstrip("#")
    r = int(hex_color[0:2], 16) / 255.0
    g = int(hex_color[2:4], 16) / 255.0
    b = int(hex_color[4:6], 16) / 255.0

    h, s, v = colorsys.rgb_to_hsv(r, g, b)
    h_cv = int(h * 179)
    s_cv = int(s * 255)
    v_cv = int(v * 255)

    lower = np.array(
        [
            [
                max(0, h_cv - tolerance),
                max(0, s_cv - tolerance * 2),
                max(0, v_cv - tolerance * 2),
            ]
        ]
    )
    upper = np.array(
        [
            [
                min(179, h_cv + tolerance),
                min(255, s_cv + tolerance * 2),
                min(255, v_cv + tolerance * 2),
            ]
        ]
    )
    return lower, upper


def quat_to_yaw(qw: float, qx: float, qy: float, qz: float) -> float:
    siny_cosp = 2 * (qw * qz + qx * qy)
    cosy_cosp = 1 - 2 * (qy * qy + qz * qz)
    return math.atan2(siny_cosp, cosy_cosp)


def body_to_ned_velocity(
    vx_body: float, vy_body: float, yaw_rad: float
) -> tuple[float, float]:
    vn = vx_body * math.cos(yaw_rad) - vy_body * math.sin(yaw_rad)
    ve = vx_body * math.sin(yaw_rad) + vy_body * math.cos(yaw_rad)
    return vn, ve


def ned_velocity_to_body(vn: float, ve: float, yaw_rad: float) -> tuple[float, float]:
    vx_body = vn * math.cos(yaw_rad) + ve * math.sin(yaw_rad)
    vy_body = -vn * math.sin(yaw_rad) + ve * math.cos(yaw_rad)
    return vx_body, vy_body


def bearing_to_yaw_delta(target_bearing_rad: float, current_yaw_rad: float) -> float:
    delta = target_bearing_rad - current_yaw_rad
    return (delta + math.pi) % (2 * math.pi) - math.pi


def pixel_offset_to_bearing(offset_px: float, focal_px: float) -> float:
    return math.atan2(offset_px, focal_px)


def estimate_focal_from_track(
    pixel_width_px: float, real_width_m: float, distance_m: float
) -> float:
    return pixel_width_px * distance_m / real_width_m
