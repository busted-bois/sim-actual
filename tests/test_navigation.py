from simulator.navigation import (
    active_gate,
    bearing_error_ned,
    distance_to_gate,
    yaw_from_state,
)


def test_active_gate_returns_target():
    data = {
        "track_gates": [
            {"gate_id": 0, "position_ned": (0.0, 0.0, -5.0)},
            {"gate_id": 1, "position_ned": (10.0, 0.0, -5.0)},
        ],
        "race_status": {"active_gate_index": 1},
    }
    gate = active_gate(data)
    assert gate["gate_id"] == 1


def test_active_gate_none_when_missing():
    assert active_gate({}) is None
    assert (
        active_gate({"track_gates": [{}], "race_status": {"active_gate_index": 5}})
        is None
    )


def test_bearing_error_ned():
    odometry = {
        "x": 0.0,
        "y": 0.0,
        "z": -5.0,
        "qx": 0.0,
        "qy": 0.0,
        "qz": 0.0,
        "qw": 1.0,
    }
    gate = {"position_ned": (10.0, 0.0, -5.0)}
    err = bearing_error_ned(odometry, gate, attitude={"yaw": 0.0})
    assert abs(err) < 0.01


def test_bearing_error_positive_when_gate_to_right():
    odometry = {
        "x": 0.0,
        "y": 0.0,
        "z": -5.0,
        "qx": 0.0,
        "qy": 0.0,
        "qz": 0.0,
        "qw": 1.0,
    }
    gate = {"position_ned": (0.0, 10.0, -5.0)}
    err = bearing_error_ned(odometry, gate, attitude={"yaw": 0.0})
    assert err > 0.5


def test_distance_to_gate():
    odometry = {"x": 0.0, "y": 0.0, "z": -5.0}
    gate = {"position_ned": (3.0, 4.0, -5.0)}
    assert distance_to_gate(odometry, gate) == 5.0


def test_yaw_from_quaternion():
    odometry = {"qx": 0.0, "qy": 0.0, "qz": 0.0, "qw": 1.0}
    assert abs(yaw_from_state(odometry)) < 1e-6
