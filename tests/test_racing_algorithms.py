import math

from simulator.flight_control import (
    perception_aware_speed,
    proportional_navigation_yaw,
    racing_command,
    velocity_mpc_body,
)
from simulator.racing_planner import gate_crossing_point, precompute_racing_path


def _gate(x, y, z=-5.0, yaw=0.0):
    half = math.sin(yaw / 2.0)
    return {
        "gate_id": 0,
        "position_ned": (x, y, z),
        "orientation_ned": (math.cos(yaw / 2.0), 0.0, 0.0, half),
    }


def test_gate_crossing_offsets_toward_next_gate():
    g0 = _gate(0.0, 0.0)
    g1 = _gate(20.0, 0.0)
    cx, cy, cz = gate_crossing_point(g0, g1, None)
    assert cx > 0.0
    assert cy == 0.0
    assert cz == -5.0


def test_precompute_racing_path_length():
    gates = [_gate(0.0, 0.0), _gate(20.0, 0.0), _gate(40.0, 5.0)]
    path = precompute_racing_path(gates)
    assert len(path) == 3


def test_perception_aware_speed_slow_on_turn():
    straight = perception_aware_speed(0.0, True, 0.05)
    turn = perception_aware_speed(math.radians(70.0), True, 0.05)
    assert straight > turn


def test_proportional_navigation_reacts_to_los_rate():
    yaw = proportional_navigation_yaw(0.0, 0.2, 0.02, 8.0)
    assert abs(yaw) > 0.0


def test_racing_command_returns_body_velocity():
    pose = {"x": 0.0, "y": 0.0, "z": -5.0, "vx": 0.0, "vy": 0.0, "yaw": 0.0}
    gate = _gate(10.0, 0.0)
    target = (8.0, 0.0, -5.0)
    cmd = racing_command(
        pose,
        target,
        gate,
        {"detected": True, "nx": 0.0, "ny": 0.0, "r_frac": 0.05},
        None,
        0.02,
    )
    assert cmd["vx"] > 0.0


def test_velocity_mpc_body_points_toward_target():
    pose = {"x": 0.0, "y": 0.0, "z": -5.0, "vx": 0.0, "vy": 0.0, "yaw": 0.0}
    vx, vy = velocity_mpc_body(pose, (10.0, 0.0, -5.0), 8.0)
    assert vx > 0.0
    assert abs(vy) < 0.5
