from unittest.mock import patch

from simulator.preflight import (
    COUNTDOWN_SCHEDULED_THRESHOLD_MS,
    RACE_COUNTDOWN_MS,
    RaceGoLatch,
    latch_race_go_boot_ms,
    poll_race_go,
    race_finished,
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
    assert go == 5000
    assert branch == "countdown"


def test_latch_countdown_start_equal():
    go, branch = latch_race_go_boot_ms(5000, 5000)
    assert go == 5000
    assert branch == "at_go"


def test_latch_threshold_boundary():
    race_start = 5000 + COUNTDOWN_SCHEDULED_THRESHOLD_MS + 1
    go, branch = latch_race_go_boot_ms(5000, race_start)
    assert branch == "scheduled"
    assert go == race_start


def test_latch_restart_scheduled_go():
    go, branch = latch_race_go_boot_ms(386, 3293, is_restart=True)
    assert go == 3293 + RACE_COUNTDOWN_MS
    assert branch == "restart_scheduled"


def test_latch_restart_at_go():
    go, branch = latch_race_go_boot_ms(6293, 3293, is_restart=True)
    assert go == 3293 + RACE_COUNTDOWN_MS
    assert branch == "restart_at_go"


def test_poll_race_go_at_timer_zero_restart():
    latch = RaceGoLatch()
    latch.reset_for_arm(300, is_restart=True)
    data = {
        "race_status": {
            "sim_boot_time_ms": 6293,
            "race_start_boot_time_ms": 3293,
        }
    }
    allowed, go_boot = poll_race_go(data, latch)
    assert go_boot == 3293 + RACE_COUNTDOWN_MS
    assert allowed is True


def test_race_go_latch_deferred_until_sim_advances():
    latch = RaceGoLatch()
    latch.reset_for_arm(386, is_restart=True)
    go, branch = latch.try_latch(386, 3293)
    assert go is None
    go, branch = latch.try_latch(500, 3293)
    assert go == 3293 + RACE_COUNTDOWN_MS
    assert branch == "restart_scheduled"


def test_race_go_latch_delta_heuristic_on_first_run():
    latch = RaceGoLatch()
    latch.reset_for_arm(50_000)
    go, branch = latch.try_latch(50_001, 80_000)
    assert go == 80_000
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


def test_race_go_allowed_restart_blocks_wrong_latch():
    data = {
        "race_status": {
            "sim_boot_time_ms": 3488,
            "race_start_boot_time_ms": 3289,
        }
    }
    assert race_go_allowed(data, go_boot_ms=3289, is_restart=True) is False
    assert race_go_allowed(data, go_boot_ms=6289, is_restart=True) is False
    data["race_status"]["sim_boot_time_ms"] = 6289
    assert race_go_allowed(data, go_boot_ms=6289, is_restart=True) is True


def test_race_go_not_allowed_during_restart_countdown():
    latch = RaceGoLatch()
    latch.reset_for_arm(6, is_restart=True)
    data = {
        "race_status": {
            "sim_boot_time_ms": 3488,
            "race_start_boot_time_ms": 3289,
        }
    }
    allowed, go_boot = poll_race_go(data, latch)
    assert go_boot == 3289 + RACE_COUNTDOWN_MS
    assert allowed is False


def test_race_go_not_allowed_during_countdown_after_latch():
    data = {
        "race_status": {
            "sim_boot_time_ms": 4980,
            "race_start_boot_time_ms": 5000,
        }
    }
    go_boot_ms, _ = latch_race_go_boot_ms(4980, 5000)
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
    assert data["_latched_go_boot_ms"] == 4000


def test_wait_for_race_go_succeeds_countdown_latch():
    data = {
        "race_status": {
            "sim_boot_time_ms": 8000,
            "race_start_boot_time_ms": 5000,
        }
    }
    assert wait_for_race_go(data, timeout_s=1.0) is True
    assert data["_latched_go_boot_ms"] == 5000


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


def test_race_finished_when_finish_ns_valid():
    assert race_finished({"race_status": {"race_finish_time_ns": 0}}) is True
    assert race_finished({"race_status": {"race_finish_time_ns": 9_000_000}}) is True


def test_race_not_finished_when_ongoing():
    assert race_finished({}) is False
    assert race_finished({"race_status": {"race_finish_time_ns": -1}}) is False
    assert race_finished({"race_status": {}}) is False
