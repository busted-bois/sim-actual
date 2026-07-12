import math
import colorsys
import numpy as np


def hex_to_hsv_lower_upper(
    hex_color: str, tolerance: int
) -> tuple[np.ndarray, np.ndarray]:
    """Convert hex color to HSV lower/upper bounds for cv2.inRange.

    Args:
        hex_color: Color in #RRGGBB format
        tolerance: HSV tolerance for matching

    Returns:
        Tuple of (lower, upper) numpy arrays, both shape (1,3) for cv2.inRange
        Note: Does NOT handle hue wraparound - caller must handle when lower[0] > upper[0]
    """
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

    return (lower, upper)


def quat_to_yaw(qw: float, qx: float, qy: float, qz: float) -> float:
    """Extract yaw from quaternion using aerospace sequence.

    Args:
        qw: Quaternion w component
        qx: Quaternion x component
        qy: Quaternion y component
        qz: Quaternion z component

    Returns:
        Yaw angle in radians
    """
    siny_cosp = 2 * (qw * qz + qx * qy)
    cosy_cosp = 1 - 2 * (qy * qy + qz * qz)
    yaw = math.atan2(siny_cosp, cosy_cosp)

    return yaw


def body_to_ned_velocity(
    vx_body: float, vy_body: float, yaw_rad: float
) -> tuple[float, float]:
    """Convert body-frame velocity to NED velocity.

    Body-forward aligns with yaw direction in NED frame.

    Args:
        vx_body: Forward velocity in body frame (m/s)
        vy_body: Lateral velocity in body frame (m/s)
        yaw_rad: Vehicle yaw angle in radians (NED frame)

    Returns:
        Tuple of (vn, ve) - North and East velocities in NED frame (m/s)
    """
    vn = vx_body * math.cos(yaw_rad) - vy_body * math.sin(yaw_rad)
    ve = vx_body * math.sin(yaw_rad) + vy_body * math.cos(yaw_rad)

    return (vn, ve)


def ned_velocity_to_body(vn: float, ve: float, yaw_rad: float) -> tuple[float, float]:
    """Convert NED velocity to body-frame velocity.

    Inverse of body_to_ned_velocity.

    Args:
        vn: North velocity in NED frame (m/s)
        ve: East velocity in NED frame (m/s)
        yaw_rad: Vehicle yaw angle in radians (NED frame)

    Returns:
        Tuple of (vx_body, vy_body) - Forward and lateral velocities in body frame (m/s)
    """
    vx_body = vn * math.cos(yaw_rad) + ve * math.sin(yaw_rad)
    vy_body = -vn * math.sin(yaw_rad) + ve * math.cos(yaw_rad)

    return (vx_body, vy_body)


def bearing_to_yaw_delta(target_bearing_rad: float, current_yaw_rad: float) -> float:
    """Calculate shortest angular difference from current yaw to target bearing.

    Args:
        target_bearing_rad: Target bearing in radians
        current_yaw_rad: Current yaw angle in radians

    Returns:
        Angular difference in radians, wrapped to [-π, π]
    """
    delta = target_bearing_rad - current_yaw_rad
    delta = (delta + math.pi) % (2 * math.pi) - math.pi

    return delta


def pixel_offset_to_bearing(offset_px: float, focal_px: float) -> float:
    """Convert pixel offset to bearing angle using small-angle approximation.

    Args:
        offset_px: Pixel offset from image center (positive = right)
        focal_px: Focal length in pixels

    Returns:
        Bearing angle in radians
    """
    bearing = math.atan2(offset_px, focal_px)

    return bearing


def estimate_focal_from_track(
    pixel_width_px: float, real_width_m: float, distance_m: float
) -> float:
    """Estimate focal length from object tracking measurements.

    Args:
        pixel_width_px: Object width in pixels
        real_width_m: Actual object width in meters
        distance_m: Distance to object in meters

    Returns:
        Focal length in pixels
    """
    focal = pixel_width_px * distance_m / real_width_m

    return focal
