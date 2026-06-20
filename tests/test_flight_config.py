import os
from unittest.mock import patch

from simulator.flight_config import (
    CAM_HEIGHT,
    CAM_TILT_UP_DEG,
    CAM_WIDTH,
    CONTROL_HZ,
    GATE_INNER_MM,
    GATE_OUTER_MM,
    resolve_auto_reset_on_collision,
)


def test_agp_camera_intrinsics():
    assert CAM_WIDTH == 640
    assert CAM_HEIGHT == 360
    assert CAM_TILT_UP_DEG == 20.0


def test_agp_gate_dimensions():
    assert GATE_OUTER_MM == 2700.0
    assert GATE_INNER_MM == 1500.0


def test_control_hz_within_spec():
    assert CONTROL_HZ < 100


def test_collision_reset_cli_overrides_env():
    with patch.dict(os.environ, {"AUTO_RESET_ON_COLLISION": "1"}):
        assert resolve_auto_reset_on_collision(False) is False
        assert resolve_auto_reset_on_collision(True) is True


def test_collision_reset_from_env():
    with patch.dict(os.environ, {"AUTO_RESET_ON_COLLISION": "true"}):
        assert resolve_auto_reset_on_collision() is True
    with patch.dict(os.environ, {}, clear=True):
        assert resolve_auto_reset_on_collision() is False
