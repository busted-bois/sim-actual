import math

from simulator.tracking.camera import (
    bearing_local_yaw_rad,
    normalized_to_pixel,
    pixel_ray_to_body_ned,
)


def test_pixel_ray_compensates_camera_pitch():
    ray = pixel_ray_to_body_ned(0.0, 0.0, pitch_up_degrees=20.0)
    assert ray[0] > 0.9


def test_bearing_positive_for_right_offset():
    bearing = bearing_local_yaw_rad(0.5, 0.0, pitch_up_degrees=20.0)
    assert bearing > 0.0


def test_normalized_to_pixel_center():
    x, y = normalized_to_pixel(0.0, 0.0, 640, 360)
    assert x == 320
    assert y == 180


def test_bearing_near_zero_for_centered_target():
    bearing = bearing_local_yaw_rad(0.0, 0.0, pitch_up_degrees=20.0)
    assert abs(bearing) < math.radians(25.0)
