import math

from simulator.flight_config import CAM_TILT_UP_DEG


def _pitch_compensation_matrix(pitch_up_degrees):
    pitch_rad = math.radians(-pitch_up_degrees)
    cp = math.cos(pitch_rad)
    sp = math.sin(pitch_rad)
    return (
        (1.0, 0.0, 0.0),
        (0.0, cp, -sp),
        (0.0, sp, cp),
    )


def _mat_vec_mul(matrix, vec):
    return tuple(sum(row[i] * vec[i] for i in range(3)) for row in matrix)


def pixel_ray_to_body_ned(nx, ny, pitch_up_degrees=CAM_TILT_UP_DEG):
    """Map normalized pixel offset to a unit ray in MAV_FRAME_BODY_NED."""
    fx = 1.0
    fy = 1.0
    ray_cam = (fx, nx * fx, ny * fy)
    norm = math.sqrt(sum(v * v for v in ray_cam))
    if norm <= 1e-9:
        return (1.0, 0.0, 0.0)
    ray_cam = tuple(v / norm for v in ray_cam)
    rot = _pitch_compensation_matrix(pitch_up_degrees)
    return _mat_vec_mul(rot, ray_cam)


def bearing_local_yaw_rad(nx, ny, pitch_up_degrees=CAM_TILT_UP_DEG):
    ray = pixel_ray_to_body_ned(nx, ny, pitch_up_degrees)
    return math.atan2(ray[1], ray[0])


def normalized_to_pixel(nx, ny, width, height):
    x = int((nx * 0.5 + 0.5) * width)
    y = int((ny * 0.5 + 0.5) * height)
    return x, y
