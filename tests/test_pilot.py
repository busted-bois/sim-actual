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
    "sim_boot_time_ms": 7000,
    "race_start_boot_time_ms": 7000,
}


def _pilot(data):
    controller = MagicMock()
    return Pilot(controller, data), controller


def _prime_countdown_go(pilot, go_boot_ms=7000):
    """Latch GO time as if countdown started at go_boot_ms - 3000."""
    countdown_start = go_boot_ms - 3000
    pilot._session_armed = True
    pilot._last_race_start_boot_ms = -1
    pilot._go_boot_ms = go_boot_ms
    pilot._awaiting_race_go = False


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
    pilot._session_armed = True
    pilot._go_boot_ms = 7000
    pilot.tick()
    kwargs = controller.set_attitude_rates.call_args.kwargs
    assert kwargs["pitch_rate"] == 0.0
    assert kwargs["thrust"] == HOVER_THRUST


def test_tick_hovers_during_countdown_from_countdown_start():
    pilot, controller = _pilot(
        {
            "armed": True,
            "track_gates": [{"gate_id": 0, "position_ned": (10.0, 0.0, -5.0)}],
            "race_status": {
                "active_gate_index": 0,
                "sim_boot_time_ms": 4980,
                "race_start_boot_time_ms": 5000,
            },
        }
    )
    pilot._session_armed = True
    pilot._go_boot_ms = 5000
    pilot.tick()
    kwargs = controller.set_attitude_rates.call_args.kwargs
    assert kwargs["pitch_rate"] == 0.0


def test_tick_fail_closed_without_latched_go():
    pilot, controller = _pilot(
        {
            "armed": True,
            "track_gates": [{"gate_id": 0, "position_ned": (10.0, 0.0, -5.0)}],
            "race_status": {
                "active_gate_index": 0,
                "sim_boot_time_ms": 9000,
                "race_start_boot_time_ms": 5000,
            },
        }
    )
    pilot._session_armed = True
    pilot._awaiting_race_go = True
    pilot._last_race_start_boot_ms = 5000
    pilot._race_go_latch.last_race_start = 5000
    pilot.tick()
    kwargs = controller.set_attitude_rates.call_args.kwargs
    assert kwargs["pitch_rate"] == 0.0
    assert pilot._go_boot_ms is None


def test_consume_main_latch():
    pilot, controller = _pilot(
        {
            "armed": True,
            "track_gates": [{"gate_id": 0, "position_ned": (10.0, 0.0, -5.0)}],
            "race_status": _RACE_GO,
            "_latched_go_boot_ms": 7000,
        }
    )
    pilot.tick()
    assert pilot._go_boot_ms == 7000
    assert pilot._session_armed is True
    kwargs = controller.set_attitude_rates.call_args.kwargs
    assert kwargs["pitch_rate"] == CRUISE_PITCH_RATE


def test_tick_cruise_when_ready():
    pilot, controller = _pilot(
        {
            "armed": True,
            "track_gates": [{"gate_id": 0, "position_ned": (10.0, 0.0, -5.0)}],
            "race_status": _RACE_GO,
        }
    )
    _prime_countdown_go(pilot)
    pilot.data["race_status"] = _RACE_GO
    controller.reset_mock()
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
    _prime_countdown_go(pilot)
    pilot.data["race_status"] = _RACE_GO
    controller.reset_mock()
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
    _prime_countdown_go(pilot)
    pilot.data["race_status"] = _RACE_GO
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
    _prime_countdown_go(pilot)
    pilot.data["race_status"] = _RACE_GO
    pilot.tick()
    kwargs = controller.set_attitude_rates.call_args.kwargs
    assert kwargs["yaw_rate"] > 0.0


def test_arms_before_first_countdown():
    pilot, controller = _pilot(
        {
            "track_gates": [{"gate_id": 0}],
            "race_status": {
                "active_gate_index": 0,
                "sim_boot_time_ms": 100,
                "race_start_boot_time_ms": -1,
            },
        }
    )
    pilot.tick()
    controller.arm.assert_called_once()
    assert pilot._session_armed is True


def test_reams_after_sim_restart_countdown_reset():
    pilot, controller = _pilot(
        {
            "armed": True,
            "track_gates": [{"gate_id": 0, "position_ned": (10.0, 0.0, -5.0)}],
            "race_status": _RACE_GO,
        }
    )
    pilot.tick()
    controller.arm.reset_mock()

    pilot.data["armed"] = False
    pilot.data["race_status"] = {
        "active_gate_index": 0,
        "sim_boot_time_ms": 200,
        "race_start_boot_time_ms": -1,
    }
    pilot.tick()
    controller.arm.assert_called_once()


def test_no_false_restart_on_sim_boot_drop_before_go():
    pilot, controller = _pilot(
        {
            "armed": True,
            "track_gates": [{"gate_id": 0, "position_ned": (10.0, 0.0, -5.0)}],
            "race_status": {
                "active_gate_index": 0,
                "sim_boot_time_ms": 454015,
                "race_start_boot_time_ms": 453993,
            },
            "_latched_go_boot_ms": 453993,
        }
    )
    pilot.tick()
    assert pilot._go_boot_ms == 453993
    assert pilot._protect_initial_go is True

    pilot.data["race_status"] = {
        "active_gate_index": 0,
        "sim_boot_time_ms": 386,
        "race_start_boot_time_ms": 3293,
    }
    pilot.tick()
    assert pilot._go_boot_ms == 453993
    assert pilot._awaiting_race_go is False


def test_restart_waits_for_scheduled_go():
    pilot, controller = _pilot(
        {
            "armed": True,
            "track_gates": [{"gate_id": 0, "position_ned": (10.0, 0.0, -5.0)}],
            "race_status": _RACE_GO,
            "_latched_go_boot_ms": 7000,
        }
    )
    pilot.tick()
    pilot._passed_go = True
    pilot._last_sim_boot_ms = 8000

    pilot.data["armed"] = False
    pilot.data["race_status"] = {
        "active_gate_index": 0,
        "sim_boot_time_ms": 300,
        "race_start_boot_time_ms": -1,
    }
    pilot.tick()
    assert pilot._awaiting_race_go is True
    assert pilot._go_boot_ms is None

    pilot.data["armed"] = True
    pilot.data["race_status"] = {
        "active_gate_index": 0,
        "sim_boot_time_ms": 500,
        "race_start_boot_time_ms": 3293,
    }
    pilot.tick()
    assert pilot._awaiting_race_go is True
    assert pilot._go_boot_ms is None
    assert controller.set_attitude_rates.call_args.kwargs["pitch_rate"] == 0.0

    pilot.data["race_status"] = {
        "active_gate_index": 0,
        "sim_boot_time_ms": 3293,
        "race_start_boot_time_ms": 3293,
    }
    pilot.tick()
    assert pilot._awaiting_race_go is False
    assert pilot._go_boot_ms == 3293
    assert (
        controller.set_attitude_rates.call_args.kwargs["pitch_rate"]
        == CRUISE_PITCH_RATE
    )


def test_flies_again_after_second_race_go():
    pilot, controller = _pilot(
        {
            "armed": True,
            "track_gates": [{"gate_id": 0, "position_ned": (10.0, 0.0, -5.0)}],
            "race_status": _RACE_GO,
            "_latched_go_boot_ms": 7000,
        }
    )
    pilot.tick()
    pilot._passed_go = True
    pilot._last_sim_boot_ms = 8000
    controller.reset_mock()

    pilot.data["armed"] = False
    pilot.data["race_status"] = {
        "active_gate_index": 0,
        "sim_boot_time_ms": 300,
        "race_start_boot_time_ms": -1,
    }
    pilot.tick()
    controller.arm.assert_called_once()

    pilot.data["armed"] = True
    pilot.data["race_status"] = {
        "active_gate_index": 0,
        "sim_boot_time_ms": 386,
        "race_start_boot_time_ms": 3293,
    }
    pilot.tick()
    kwargs = controller.set_attitude_rates.call_args.kwargs
    assert kwargs["pitch_rate"] == 0.0
    assert pilot._awaiting_race_go is True

    pilot.data["race_status"] = {
        "active_gate_index": 0,
        "sim_boot_time_ms": 3293,
        "race_start_boot_time_ms": 3293,
    }
    pilot.tick()
    kwargs = controller.set_attitude_rates.call_args.kwargs
    assert kwargs["pitch_rate"] == CRUISE_PITCH_RATE
    assert pilot._go_boot_ms == 3293


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
    _prime_countdown_go(pilot)
    pilot.data["race_status"] = _RACE_GO
    pilot.tick()
    thrust = controller.set_attitude_rates.call_args.kwargs["thrust"]
    assert thrust < CRUISE_THRUST
