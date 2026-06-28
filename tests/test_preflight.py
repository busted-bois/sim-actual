import unittest
from unittest.mock import patch

from simulator.preflight import (
    COUNTDOWN_SCHEDULED_THRESHOLD_MS,
    RACE_COUNTDOWN_MS,
    RaceGoLatch,
    is_restart_arm_context,
    latch_race_go_boot_ms,
    poll_race_go,
    race_finished,
    race_go_allowed,
    wait_for_fresh_track,
    wait_for_race_go,
    wait_for_race_start,
    wait_for_track,
)


class PreflightRaceGoTests(unittest.TestCase):
    def test_latch_scheduled_go(self):
        go, branch = latch_race_go_boot_ms(5000, 8000)
        self.assertEqual(go, 8000)
        self.assertEqual(branch, "scheduled")

    def test_latch_countdown_start_future(self):
        go, branch = latch_race_go_boot_ms(4980, 5000)
        self.assertEqual(go, 5000)
        self.assertEqual(branch, "countdown")

    def test_race_go_not_allowed_during_countdown_after_latch(self):
        data = {
            "race_status": {
                "sim_boot_time_ms": 4980,
                "race_start_boot_time_ms": 5000,
            }
        }
        go_boot_ms, _ = latch_race_go_boot_ms(4980, 5000)
        self.assertFalse(race_go_allowed(data, go_boot_ms=go_boot_ms))

    def test_race_go_allowed_at_go(self):
        data = {
            "race_status": {
                "sim_boot_time_ms": 7000,
                "race_start_boot_time_ms": 7000,
            }
        }
        self.assertTrue(race_go_allowed(data, go_boot_ms=7000))

    def test_poll_race_go_restart_blocks_until_countdown_done(self):
        latch = RaceGoLatch()
        latch.reset_for_arm(6, is_restart=True)
        data = {
            "race_status": {
                "sim_boot_time_ms": 3488,
                "race_start_boot_time_ms": 3289,
            }
        }
        allowed, go_boot = poll_race_go(data, latch)
        self.assertEqual(go_boot, 3289 + RACE_COUNTDOWN_MS)
        self.assertFalse(allowed)

    def test_latch_succeeds_after_arm_boot_block(self):
        latch = RaceGoLatch()
        latch.reset_for_arm(10000, is_restart=False)
        data = {
            "race_status": {
                "sim_boot_time_ms": 10000,
                "race_start_boot_time_ms": 12000,
            }
        }
        allowed, go_boot = poll_race_go(data, latch)
        self.assertFalse(allowed)
        self.assertIsNone(go_boot)
        self.assertIsNone(latch.go_boot_ms)

        data["race_status"]["sim_boot_time_ms"] = 10001
        allowed, go_boot = poll_race_go(data, latch)
        self.assertIsNotNone(latch.go_boot_ms)
        self.assertEqual(go_boot, 12000)

    def test_wait_for_track_succeeds(self):
        data = {"track_gates": [{"gate_id": 0}]}
        self.assertTrue(wait_for_track(data, timeout_s=1.0))

    def test_wait_for_race_go_succeeds_countdown_latch(self):
        data = {
            "race_status": {
                "sim_boot_time_ms": 8000,
                "race_start_boot_time_ms": 5000,
            }
        }
        self.assertTrue(wait_for_race_go(data, timeout_s=1.0))
        self.assertEqual(data["_latched_go_boot_ms"], 5000)

    def test_race_finished_when_finish_ns_valid(self):
        self.assertTrue(race_finished({"race_status": {"race_finish_time_ns": 0}}))

    def test_race_not_finished_when_ongoing(self):
        self.assertFalse(race_finished({"race_status": {"race_finish_time_ns": -1}}))

    def test_latch_threshold_boundary(self):
        race_start = 5000 + COUNTDOWN_SCHEDULED_THRESHOLD_MS + 1
        go, branch = latch_race_go_boot_ms(5000, race_start)
        self.assertEqual(branch, "scheduled")
        self.assertEqual(go, race_start)

    def test_restart_context_false_when_boot_zero(self):
        self.assertFalse(is_restart_arm_context(0))
        self.assertFalse(is_restart_arm_context(None))

    def test_wait_for_race_start_succeeds(self):
        data = {"race_status": {"race_start_boot_time_ms": 1000}}
        self.assertTrue(wait_for_race_start(data, timeout_s=0.5))


class PreflightTimeoutTests(unittest.TestCase):
    def test_wait_for_track_times_out(self):
        with patch("simulator.preflight.PREFLIGHT_POLL_S", 0.01):
            self.assertFalse(wait_for_track({}, timeout_s=0.05))

    def test_wait_for_fresh_track_clears_stale_and_succeeds(self):
        data = {"track_gates": [{"stale": True}]}
        with patch("simulator.preflight.PREFLIGHT_POLL_S", 0.01):
            self.assertFalse(wait_for_fresh_track(data, timeout_s=0.02))
        self.assertNotIn("track_gates", data)

        data = {}
        polls = {"n": 0}

        def inject_burst(_delay):
            polls["n"] += 1
            if polls["n"] == 1:
                data["track_gates"] = [{"gate_id": 0}]

        with patch("simulator.preflight.PREFLIGHT_POLL_S", 0.01):
            with patch("simulator.preflight.time.sleep", side_effect=inject_burst):
                self.assertTrue(wait_for_fresh_track(data, timeout_s=1.0))

    def test_wait_for_race_go_times_out(self):
        data = {
            "race_status": {
                "sim_boot_time_ms": 1000,
                "race_start_boot_time_ms": 5000,
            }
        }
        with patch("simulator.preflight.RACE_GO_POLL_S", 0.01):
            self.assertFalse(wait_for_race_go(data, timeout_s=0.05))


if __name__ == "__main__":
    unittest.main()
