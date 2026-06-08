import math

from simulator.math_util import clamp, normalize_angle


def test_clamp_limits_value():
    assert clamp(5.0, 0.0, 3.0) == 3.0
    assert clamp(-1.0, 0.0, 3.0) == 0.0


def test_normalize_angle_wraps():
    assert abs(normalize_angle(math.pi + 0.5) + math.pi - 0.5) < 1e-9
