from unittest.mock import patch

from simulator.preflight import (
    COUNTDOWN_SCHEDULED_THRESHOLD_MS,
    RACE_COUNTDOWN_MS,
    RaceGoLatch,
    arm_relative_go_boot_ms,
    latch_race_go_boot_ms,
    race_go_allowed,
    wait_for_race_go,
    wait_for_ready,
    wait_for_track,
)


def test_latch_scheduled_go():
    go, branch = latch_race_go_boot_ms(5000, 8000)
    assert go == 8000
    assert branch == "scheduled"


def test_latch_countdown_start_future():
    go, branch = latch_race_go_boot_ms(4980, 5000)
    assert go == 5000 + RACE_COUNTDOWN_MS
    assert branch == "countdown"


def test_latch_countdown_start_equal():
    go, branch = latch_race_go_boot_ms(5000, 5000)
    assert go == 8000
    assert branch == "countdown"


def test_latch_threshold_boundary():
    race_start = 5000 + COUNTDOWN_SCHEDULED_THRESHOLD_MS + 1
    go, branch = latch_race_go_boot_ms(5000, race_start)
    assert branch == "scheduled"
    assert go == race_start


def test_arm_relative_go_boot_ms():
    assert arm_relative_go_boot_ms(1000) == 1000 + RACE_COUNTDOWN_MS
    assert arm_relative_go_boot_ms(None) is None


def test_latch_restart_countdown_future():
    go, branch = latch_race_go_boot_ms(386, 3293, is_restart=True)
    assert go == 6293
    assert branch == "countdown_future"


def test_race_go_latch_restart_countdown_future():
    latch = RaceGoLatch()
    latch.reset_for_arm(is_restart=True)
    go, branch = latch.try_latch(386, 3293)
    assert go == 6293
    assert branch == "countdown_future"


def test_race_go_latch_delta_heuristic_on_first_run():
    latch = RaceGoLatch()
    latch.reset_for_arm(is_restart=False)
    go, branch = latch.try_latch(5000, 8000)
    assert go == 8000
    assert branch == "scheduled"


def test_race_go_allowed_before_go():
    data = {
        "race_status": {
            "sim_boot_time_ms": 5000,
            "race_start_boot_time_ms": 7000,
        }
    }
    assert race_go_allowed(data, go_boot_ms=7000) is False


def test_race_go_allowed_at_go():
    data = {
        "race_status": {
            "sim_boot_time_ms": 7000,
            "race_start_boot_time_ms": 7000,
        }
    }
    assert race_go_allowed(data, go_boot_ms=7000) is True


def test_race_go_allowed_fail_closed_without_latch():
    data = {
        "race_status": {
            "sim_boot_time_ms": 8000,
            "race_start_boot_time_ms": 5000,
        }
    }
    assert race_go_allowed(data, go_boot_ms=None) is False


def test_race_go_allowed_when_race_not_started():
    data = {"race_status": {"sim_boot_time_ms": 5000, "race_start_boot_time_ms": -1}}
    assert race_go_allowed(data, go_boot_ms=8000) is False


def test_race_go_not_allowed_during_countdown_after_latch():
    data = {
        "race_status": {
            "sim_boot_time_ms": 6000,
            "race_start_boot_time_ms": 5000,
        }
    }
    go_boot_ms, _ = latch_race_go_boot_ms(5000, 5000)
    assert race_go_allowed(data, go_boot_ms=go_boot_ms) is False


def test_wait_for_track_succeeds():
    data = {"track_gates": [{"gate_id": 0}]}
    assert wait_for_track(data, timeout_s=1.0) is True


def test_wait_for_track_times_out():
    with patch("simulator.preflight.PREFLIGHT_POLL_S", 0.01):
        assert wait_for_track({}, timeout_s=0.05) is False


def test_wait_for_race_go_succeeds_scheduled():
    data = {
        "race_status": {
            "sim_boot_time_ms": 7000,
            "race_start_boot_time_ms": 4000,
        }
    }
    assert wait_for_race_go(data, timeout_s=1.0) is True
    assert data["_latched_go_boot_ms"] == 7000


def test_wait_for_race_go_succeeds_countdown_latch():
    data = {
        "race_status": {
            "sim_boot_time_ms": 8000,
            "race_start_boot_time_ms": 5000,
        }
    }
    assert wait_for_race_go(data, timeout_s=1.0) is True
    assert data["_latched_go_boot_ms"] == 8000


def test_wait_for_race_go_times_out():
    data = {
        "race_status": {
            "sim_boot_time_ms": 1000,
            "race_start_boot_time_ms": 5000,
        }
    }
    with patch("simulator.preflight.RACE_GO_POLL_S", 0.01):
        assert wait_for_race_go(data, timeout_s=0.05) is False


def test_wait_for_ready_alias():
    data = {"track_gates": [{"gate_id": 0}]}
    assert wait_for_ready(data, timeout_s=1.0) is True


def test_probe_udp_port_roundtrip():
    from simulator.preflight import probe_udp_port

    ok, err = probe_udp_port(port=0)
    assert ok is True
    assert err is None
