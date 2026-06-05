import time
from unittest.mock import patch

from simulator.preflight import (
    ensure_countdown_fallback,
    race_go_allowed,
    wait_for_ready,
)


def test_race_go_allowed_before_go():
    data = {
        "race_status": {
            "sim_boot_time_ms": 5000,
            "race_start_boot_time_ms": 7000,
        }
    }
    assert race_go_allowed(data) is False


def test_race_go_allowed_after_go():
    data = {
        "race_status": {
            "sim_boot_time_ms": 8000,
            "race_start_boot_time_ms": 7000,
        }
    }
    assert race_go_allowed(data) is True


def test_race_go_allowed_when_race_not_started():
    data = {"race_status": {"sim_boot_time_ms": 5000, "race_start_boot_time_ms": -1}}
    assert race_go_allowed(data) is False


def test_race_go_allowed_fallback_after_countdown():
    data = {}
    ensure_countdown_fallback(data)
    data["_fallback_race_go_at"] = time.monotonic() - 0.01
    assert race_go_allowed(data) is True


def test_race_go_allowed_without_race_status_before_fallback():
    assert race_go_allowed({}) is False


def test_wait_for_ready_succeeds_when_data_present():
    data = {"armed": True, "track_gates": [{"gate_id": 0}]}
    assert wait_for_ready(data, timeout_s=1.0) is True


def test_wait_for_ready_times_out():
    data = {"armed": False}
    with patch("simulator.preflight.PREFLIGHT_POLL_S", 0.01):
        start = time.time()
        assert wait_for_ready(data, timeout_s=0.05) is False
        assert time.time() - start >= 0.05
