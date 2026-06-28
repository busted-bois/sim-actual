import unittest
from unittest.mock import patch

from simulator.preflight import (
    COUNTDOWN_SCHEDULED_THRESHOLD_MS,
    GO_POST_HOLD_S,
    GO_POST_MARGIN_MS,
    RACE_COUNTDOWN_MS,
    RaceGoLatch,
    is_restart_arm_context,
    latch_race_go_boot_ms,
    poll_race_go,
    race_finished,
    race_go_allowed,
    race_go_already_passed,
    wait_after_race_go,
    wait_for_fresh_race_start,
    wait_for_fresh_track,
    wait_for_race_go,
    wait_for_race_start,
    wait_for_track,
    wait_for_visual_go,
)


class PreflightRaceGoTests(unittest.TestCase):
    def test_latch_scheduled_go(self):
        go, branch = latch_race_go_boot_ms(5000, 8000)
        self.assertEqual(go, 8000)
        self.assertEqual(branch, "scheduled")

    def test_latch_countdown_start_future(self):
        go, branch = latch_race_go_boot_ms(4980, 5000)
        self.assertEqual(go, 5000 + RACE_COUNTDOWN_MS)
        self.assertEqual(branch, "countdown")

    def test_latch_imminent_countdown_go(self):
        go, branch = latch_race_go_boot_ms(1_194_695, 1_194_661)
        self.assertEqual(go, 1_194_661 + RACE_COUNTDOWN_MS)
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
                "sim_boot_time_ms": 10000,
                "race_start_boot_time_ms": 7000,
            }
        }
        self.assertTrue(race_go_allowed(data, go_boot_ms=10000))

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
                "sim_boot_time_ms": 4980,
                "race_start_boot_time_ms": 5000,
            }
        }
        calls = {"n": 0}

        def bump(_dt):
            calls["n"] += 1
            if calls["n"] > 20:
                data["race_status"]["sim_boot_time_ms"] = (
                    5000 + RACE_COUNTDOWN_MS + GO_POST_MARGIN_MS
                )

        with patch("simulator.preflight.RACE_GO_POLL_S", 0.001):
            with patch("simulator.preflight.time.sleep", side_effect=bump):
                self.assertTrue(wait_for_race_go(data, timeout_s=1.0))
        self.assertEqual(data["_latched_go_boot_ms"], 5000 + RACE_COUNTDOWN_MS)

    def test_wait_for_race_go_rejects_instant_at_go(self):
        data = {
            "race_status": {
                "sim_boot_time_ms": 8000,
                "race_start_boot_time_ms": 5000,
            }
        }
        with patch("simulator.preflight.RACE_GO_POLL_S", 0.01):
            self.assertFalse(wait_for_race_go(data, timeout_s=0.05))

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

    def test_race_go_already_passed_after_countdown(self):
        data = {
            "race_status": {
                "sim_boot_time_ms": 8000,
                "race_start_boot_time_ms": 5000,
            }
        }
        self.assertTrue(race_go_already_passed(data))

    def test_race_go_not_passed_during_countdown(self):
        data = {
            "race_status": {
                "sim_boot_time_ms": 3500,
                "race_start_boot_time_ms": 3289,
            }
        }
        self.assertFalse(race_go_already_passed(data))

    def test_wait_for_fresh_race_start_rejects_stale(self):
        data = {
            "race_status": {
                "sim_boot_time_ms": 8000,
                "race_start_boot_time_ms": 5000,
            }
        }
        with patch("simulator.preflight.PREFLIGHT_POLL_S", 0.01):
            self.assertFalse(wait_for_fresh_race_start(data, timeout_s=0.03))

    def test_wait_for_fresh_race_start_rejects_stale_mid_countdown(self):
        data = {
            "_preflight_race_start_baseline": 1_194_661,
            "_track_sim_boot_ms": 1_200_000,
            "_track_race_start_ms": 1_194_661,
            "race_status": {
                "sim_boot_time_ms": 1_194_695,
                "race_start_boot_time_ms": 1_194_661,
            },
        }
        with patch("simulator.preflight.PREFLIGHT_POLL_S", 0.01):
            self.assertFalse(wait_for_fresh_race_start(data, timeout_s=0.03))

    def test_wait_for_fresh_race_start_accepts_live_countdown(self):
        data = {
            "_preflight_race_start_baseline": 4000,
            "_track_sim_boot_ms": 4980,
            "_track_race_start_ms": 4000,
            "race_status": {
                "sim_boot_time_ms": 4980,
                "race_start_boot_time_ms": 5000,
            },
        }
        self.assertTrue(wait_for_fresh_race_start(data, timeout_s=0.5))

    def test_wait_for_fresh_race_start_accepts_scheduled_future(self):
        data = {
            "_track_sim_boot_ms": 5000,
            "_track_race_start_ms": 5000,
            "race_status": {
                "sim_boot_time_ms": 5000,
                "race_start_boot_time_ms": 8000,
            },
        }
        self.assertTrue(wait_for_fresh_race_start(data, timeout_s=0.5))

    def test_wait_for_fresh_race_start_accepts_new_race_start(self):
        data = {
            "_preflight_race_start_baseline": 1_000_000,
            "_track_sim_boot_ms": 1_200_000,
            "_track_race_start_ms": 1_000_000,
            "race_status": {
                "sim_boot_time_ms": 1_200_010,
                "race_start_boot_time_ms": 1_200_050,
            },
        }
        self.assertTrue(wait_for_fresh_race_start(data, timeout_s=0.5))

    def test_restart_early_delta_does_not_start_early(self):
        """Regression: restart with sim_boot << race_start must wait for full countdown.

        After sim_reset, sim_boot resets to near-zero.  If we sample race_start
        before the countdown has advanced 1500 ms, the delta (race_start - sim_boot)
        exceeds COUNTDOWN_SCHEDULED_THRESHOLD_MS.  Previously this triggered the
        "scheduled" branch and set go_boot = race_start (countdown-start time),
        causing the drone to start ~3 s before actual GO.
        """
        # Restart context: sim_boot is tiny (post-reset), race_start is countdown start.
        sim_boot = 500
        race_start = 3289
        # delta = 2789 ms > COUNTDOWN_SCHEDULED_THRESHOLD_MS (1500 ms)
        # is_restart=True must override the delta heuristic.
        go, branch = latch_race_go_boot_ms(sim_boot, race_start, is_restart=True)
        self.assertEqual(go, race_start + RACE_COUNTDOWN_MS)
        self.assertNotEqual(branch, "at_go")

        # Verify race_go_allowed correctly reports NOT allowed before actual GO
        data = {
            "race_status": {
                "sim_boot_time_ms": sim_boot,
                "race_start_boot_time_ms": race_start,
            }
        }
        self.assertFalse(race_go_allowed(data, go_boot_ms=go))

    def test_wait_for_race_go_restart_early_delta_blocks(self):
        """wait_for_race_go must not fire early during restart with large delta."""
        data = {
            "race_status": {
                "sim_boot_time_ms": 500,
                "race_start_boot_time_ms": 3289,
            }
        }
        # With sim_boot=500 < RESTART_ARM_BOOT_THRESHOLD_MS, is_restart auto-detected.
        with patch("simulator.preflight.RACE_GO_POLL_S", 0.01):
            self.assertFalse(wait_for_race_go(data, timeout_s=0.05))

    def test_wait_after_race_go_holds_before_flight(self):
        class FakeController:
            def __init__(self):
                self.safe_calls = 0
                self._controls_enabled = False
                self._thrust = 0.0
                self.data = {}

            def set_controls_enabled(self, _enabled):
                pass

            def send_safe_hold(self):
                self.safe_calls += 1

        controller = FakeController()
        t0 = {"now": 100.0}

        def fake_monotonic():
            return t0["now"]

        def fake_sleep(_dt):
            t0["now"] += GO_POST_HOLD_S + 0.01

        with patch("simulator.preflight.time.monotonic", side_effect=fake_monotonic):
            with patch("simulator.preflight.time.sleep", side_effect=fake_sleep):
                self.assertTrue(wait_after_race_go(controller=controller))
        self.assertGreaterEqual(controller.safe_calls, 1)

    def test_wait_for_visual_go_fallback_when_no_countdown(self):
        data = {}
        with patch("simulator.preflight.VISUAL_GO_SEE_TIMEOUT_S", 0.01):
            with patch("simulator.preflight.PREFLIGHT_POLL_S", 0.01):
                self.assertTrue(wait_for_visual_go(data, timeout_s=0.5))

    def test_wait_for_visual_go_cleared(self):
        from simulator.countdown_detector import reset_countdown_gate

        data = {}
        reset_countdown_gate(data)
        data["_countdown_gate"]["state"] = "cleared"
        self.assertTrue(wait_for_visual_go(data, timeout_s=0.5))


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
