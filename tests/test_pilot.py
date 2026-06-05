import time
from unittest.mock import MagicMock

from simulator.pilot import (
    COLLISION_THRUST,
    CRUISE_PITCH_RATE,
    CRUISE_THRUST,
    HOVER_THRUST,
    Pilot,
)


_RACE_GO = {
    "active_gate_index": 0,
    "sim_boot_time_ms": 8000,
    "race_start_boot_time_ms": 7000,
}


def _pilot(data):
    controller = MagicMock()
    return Pilot(controller, data), controller


def test_tick_hovers_when_not_armed():
    pilot, controller = _pilot({})
    pilot.tick()
    controller.set_attitude_rates.assert_called_once()
    assert controller.set_attitude_rates.call_args.kwargs["thrust"] == HOVER_THRUST


def test_tick_hovers_when_track_gates_missing():
    pilot, controller = _pilot({"armed": True})
    pilot.tick()
    controller.set_attitude_rates.assert_called_once()
    assert controller.set_attitude_rates.call_args.kwargs["thrust"] == HOVER_THRUST


def test_tick_reduces_thrust_on_collision():
    pilot, controller = _pilot({"armed": True, "collision": {"id": 1001}})
    pilot.tick()
    controller.set_attitude_rates.assert_called_once_with(
        roll_rate=0.0, pitch_rate=0.0, yaw_rate=0.0, thrust=COLLISION_THRUST
    )


def test_tick_clears_collision_after_hold():
    pilot, controller = _pilot(
        {"armed": True, "track_gates": [{}], "collision": {"id": 1001}}
    )
    pilot._collision_hold_start = time.monotonic() - 10.0
    pilot.tick()
    assert "collision" not in pilot.data


def test_tick_hovers_during_countdown():
    pilot, controller = _pilot(
        {
            "armed": True,
            "track_gates": [{"gate_id": 0, "position_ned": (10.0, 0.0, -5.0)}],
            "race_status": {
                "active_gate_index": 0,
                "sim_boot_time_ms": 5000,
                "race_start_boot_time_ms": 7000,
            },
        }
    )
    pilot.tick()
    kwargs = controller.set_attitude_rates.call_args.kwargs
    assert kwargs["pitch_rate"] == 0.0
    assert kwargs["thrust"] == HOVER_THRUST


def test_tick_cruise_when_ready():
    pilot, controller = _pilot(
        {
            "armed": True,
            "track_gates": [{"gate_id": 0, "position_ned": (10.0, 0.0, -5.0)}],
            "race_status": _RACE_GO,
        }
    )
    pilot.tick()
    controller.set_control_mode.assert_called_once_with("attitude")
    kwargs = controller.set_attitude_rates.call_args.kwargs
    assert kwargs["pitch_rate"] == CRUISE_PITCH_RATE
    assert kwargs["thrust"] == CRUISE_THRUST


def test_tick_vision_steering_when_gate_detected():
    pilot, controller = _pilot(
        {
            "armed": True,
            "track_gates": [{"gate_id": 0}],
            "race_status": _RACE_GO,
            "camera": {"received_at": time.time()},
            "gate_target": {"detected": True, "nx": 0.5, "ny": 0.0, "r_frac": 0.05},
        }
    )
    pilot.tick()
    controller.set_control_mode.assert_called_once_with("attitude")
    kwargs = controller.set_attitude_rates.call_args.kwargs
    assert kwargs["yaw_rate"] > 0.0
    assert kwargs["pitch_rate"] < 0.0


def test_tick_ignores_stale_vision():
    pilot, controller = _pilot(
        {
            "armed": True,
            "track_gates": [{"gate_id": 0, "position_ned": (10.0, 0.0, -5.0)}],
            "race_status": _RACE_GO,
            "camera": {"received_at": time.time() - 5.0},
            "gate_target": {"detected": True, "nx": 0.5, "ny": 0.0, "r_frac": 0.05},
            "odometry": {
                "x": 0.0,
                "y": 0.0,
                "z": -5.0,
                "vz": 0.0,
                "qx": 0.0,
                "qy": 0.0,
                "qz": 0.0,
                "qw": 1.0,
            },
            "attitude": {"yaw": 0.0},
        }
    )
    pilot.tick()
    kwargs = controller.set_attitude_rates.call_args.kwargs
    assert kwargs["yaw_rate"] == 0.0
    assert kwargs["pitch_rate"] < 0.0


def test_tick_telemetry_yaw_toward_gate():
    pilot, controller = _pilot(
        {
            "armed": True,
            "track_gates": [{"gate_id": 0, "position_ned": (0.0, 20.0, -5.0)}],
            "race_status": _RACE_GO,
            "odometry": {
                "x": 0.0,
                "y": 0.0,
                "z": -5.0,
                "vz": 0.0,
                "qx": 0.0,
                "qy": 0.0,
                "qz": 0.0,
                "qw": 1.0,
            },
            "attitude": {"yaw": 0.0},
        }
    )
    pilot.tick()
    kwargs = controller.set_attitude_rates.call_args.kwargs
    assert kwargs["yaw_rate"] > 0.0


def test_altitude_pid_adjusts_thrust_with_odometry():
    pilot, controller = _pilot(
        {
            "armed": True,
            "track_gates": [{"gate_id": 0, "position_ned": (10.0, 0.0, -5.0)}],
            "race_status": _RACE_GO,
            "odometry": {
                "x": 0.0,
                "y": 0.0,
                "z": -7.0,
                "vz": 0.0,
                "qx": 0.0,
                "qy": 0.0,
                "qz": 0.0,
                "qw": 1.0,
            },
            "attitude": {"yaw": 0.0},
        }
    )
    pilot.tick()
    thrust = controller.set_attitude_rates.call_args.kwargs["thrust"]
    assert thrust < CRUISE_THRUST
