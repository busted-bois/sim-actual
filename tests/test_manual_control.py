import math
from unittest.mock import MagicMock

import pytest

from simulator import manual_control
from simulator.manual_control import (
    CONTROL_DT_S,
    HORIZONTAL_SPEED_M_S,
    PITCH_RATE_PER_M_S,
    ROLL_RATE_PER_M_S,
    VERTICAL_SPEED_M_S,
    YAW_RATE_RAD_S,
    ManualControl,
)


class _FakeKeyboard:
    def __init__(self):
        self.pressed = set()

    def is_pressed(self, key):
        return key in self.pressed


@pytest.fixture
def fake_keyboard(monkeypatch):
    kb = _FakeKeyboard()
    monkeypatch.setattr(manual_control, "keyboard", kb)
    return kb


def _manual(data=None):
    controller = MagicMock()
    controller.pilot = MagicMock()
    controller.pilot._altitude_thrust.return_value = 0.55
    return ManualControl(controller, data or {}), controller


def test_inactive_by_default(fake_keyboard):
    manual, controller = _manual()
    assert manual.tick() is False
    controller.set_attitude_rates.assert_not_called()


def test_toggle_on_edge_not_hold(fake_keyboard):
    manual, _ = _manual()
    fake_keyboard.pressed.add("n")
    manual.tick()
    assert manual.active is True
    manual.tick()  # still held: no re-toggle
    assert manual.active is True
    fake_keyboard.pressed.discard("n")
    manual.tick()
    fake_keyboard.pressed.add("n")
    manual.tick()
    assert manual.active is False


def test_toggle_on_captures_current_altitude(fake_keyboard):
    manual, _ = _manual({"odometry": {"z": -12.5, "vz": 0.0}})
    fake_keyboard.pressed.add("n")
    manual.tick()
    assert manual._hold_z == pytest.approx(-12.5)


def test_toggle_falls_back_to_local_position_ned(fake_keyboard):
    manual, _ = _manual({"local_position_ned": {"z": -8.0}})
    fake_keyboard.pressed.add("n")
    manual.tick()
    assert manual._hold_z == pytest.approx(-8.0)


def test_wasd_cardinal_and_qe_vertical(fake_keyboard):
    manual, controller = _manual({"odometry": {"z": -5.0, "vz": 0.0}})
    manual.active = True
    manual._hold_z = -5.0
    fake_keyboard.pressed.update({"w", "d", "q"})
    assert manual.tick() is True
    controller.set_control_mode.assert_called_with("attitude")
    kwargs = controller.set_attitude_rates.call_args.kwargs
    assert kwargs["pitch_rate"] == pytest.approx(-PITCH_RATE_PER_M_S * HORIZONTAL_SPEED_M_S)
    assert kwargs["roll_rate"] == pytest.approx(ROLL_RATE_PER_M_S * HORIZONTAL_SPEED_M_S)
    assert kwargs["yaw_rate"] == 0.0
    assert kwargs["thrust"] == 0.55
    assert manual._z_offset == pytest.approx(-VERTICAL_SPEED_M_S * CONTROL_DT_S)


def test_up_arrow_moves_along_camera_yaw(fake_keyboard):
    yaw = math.pi / 2  # camera facing east
    manual, controller = _manual({"attitude": {"yaw": yaw}})
    manual.active = True
    fake_keyboard.pressed.add("up")
    manual.tick()
    kwargs = controller.set_attitude_rates.call_args.kwargs
    assert kwargs["pitch_rate"] == pytest.approx(
        -PITCH_RATE_PER_M_S * HORIZONTAL_SPEED_M_S
    )
    assert kwargs["roll_rate"] == pytest.approx(0.0, abs=1e-9)


def test_left_right_arrows_set_yaw_rate(fake_keyboard):
    manual, controller = _manual()
    manual.active = True
    fake_keyboard.pressed.add("left")
    manual.tick()
    assert controller.set_attitude_rates.call_args.kwargs["yaw_rate"] == -YAW_RATE_RAD_S


def test_no_keys_hovers_like_autopilot(fake_keyboard):
    manual, controller = _manual()
    manual.active = True
    manual._hold_z = -5.0
    manual.tick()
    controller.set_control_mode.assert_called_with("attitude")
    controller.set_attitude_rates.assert_called_with(
        roll_rate=0.0,
        pitch_rate=0.0,
        yaw_rate=0.0,
        thrust=0.55,
    )
