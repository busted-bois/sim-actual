"""Pilot residual + yaw-hold tests with a fake controller (no sim, no torch)."""

import math

import numpy as np
import pytest

from simulator import pilot as pilot_mod
from simulator.config import ROUND1_GATES
from simulator.pilot import Pilot, _perp_right_horizontal
from simulator.rl_core import GuidanceResidual


class FakeController:
    def __init__(self):
        self.mode = None
        self.last_cmd = None

    def set_control_mode(self, mode):
        self.mode = mode

    def set_attitude_rates(self, roll, pitch, yaw_rate, thrust):
        self.last_cmd = (roll, pitch, yaw_rate, thrust)


def _make_pilot():
    return Pilot(FakeController(), {})


def test_pilot_init_identity_no_policy():
    p = _make_pilot()
    assert p.residual.speed_mult == 1.0
    assert p.external_residual is False
    assert p._policy is None  # no policy.pt present


def test_perp_right_horizontal():
    # down-track facing -X (south-ish): right is +Y? right of (n,e)=(-1,0) is (-e,n)=(0,-1)
    pr = _perp_right_horizontal((-1.0, 0.0, 0.0))
    assert pr[0] == pytest.approx(0.0, abs=1e-6)
    assert pr[1] == pytest.approx(-1.0, abs=1e-6)
    # facing north (1,0) -> right is east (0,1)
    pr = _perp_right_horizontal((1.0, 0.0, 0.0))
    assert pr == pytest.approx((0.0, 1.0))


def test_yaw_hold_deadband_and_sign():
    p = _make_pilot()
    appr = (-1.0, 0.0, 0.0)  # desired bearing = atan2(0,-1) = pi
    # exactly on heading -> within deadband -> zero
    assert p._yaw_hold_rate(appr, math.pi) == 0.0
    # heading error beyond deadband -> nonzero, clamped
    rate = p._yaw_hold_rate(appr, math.pi - 0.5)
    assert rate != 0.0
    assert abs(rate) <= pilot_mod.MAX_YAW_RATE


def _race_data(yaw=math.pi):
    return {
        "armed": True,
        "odometry": {
            "x": -5.0,
            "y": 0.0,
            "z": 0.0,
            "vx": 0,
            "vy": 0,
            "vz": 0,
            "qx": 0,
            "qy": 0,
            "qz": 0,
            "qw": 1,
        },
        "track_gates": ROUND1_GATES,
        "active_gate_index": 0,
        "yaw_rad": yaw,
        "attitude": {"roll": 0.0, "pitch": 0.0, "yaw": yaw},
    }


def test_tick_bare_pilot_commands_within_bounds():
    p = _make_pilot()
    p.data = _race_data()
    p.tick()
    roll, pitch, yaw_rate, thrust = p.controller.last_cmd
    assert -pilot_mod.MAX_ROLL <= roll <= pilot_mod.MAX_ROLL
    assert -pilot_mod.MAX_FWD_PITCH <= pitch <= pilot_mod.MAX_BRAKE_PITCH
    assert 0.0 <= thrust <= 1.0
    assert abs(yaw_rate) <= pilot_mod.MAX_YAW_RATE


def test_tick_with_residual_runs_and_stays_bounded():
    p = _make_pilot()
    p.external_residual = True
    p.data = _race_data()
    # aggressive residual: max speed, big lateral cut
    p.residual = GuidanceResidual(speed_mult=2.5, lookahead_m=8.0, lateral_offset_m=1.5)
    p.tick()
    roll, pitch, yaw_rate, thrust = p.controller.last_cmd
    # inner-loop safety bounds must still hold regardless of the residual
    assert -pilot_mod.MAX_ROLL <= roll <= pilot_mod.MAX_ROLL
    assert -pilot_mod.MAX_FWD_PITCH <= pitch <= pilot_mod.MAX_BRAKE_PITCH
    assert 0.0 <= thrust <= 1.0
    assert np.isfinite([roll, pitch, yaw_rate, thrust]).all()
