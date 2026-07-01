import unittest
from unittest.mock import MagicMock, patch

from simulator.auto_flight import (
    CancelListener,
    auto_flight_enabled,
    run_auto_flight_loop,
)
from simulator.preflight import (
    race_go_already_passed,
    wait_for_fresh_race_start_vq2,
    wait_for_session_ready,
)
from simulator.race_monitor import course_complete, gate1_fail, passed_first_gate


class AutoFlightTests(unittest.TestCase):
    def test_auto_flight_enabled(self):
        with patch.dict("os.environ", {"AUTO_FLIGHT": "1"}):
            self.assertTrue(auto_flight_enabled())
        with patch.dict("os.environ", {"AUTO_FLIGHT": "true"}):
            self.assertTrue(auto_flight_enabled())
        with patch.dict("os.environ", {}, clear=True):
            self.assertFalse(auto_flight_enabled())

    def test_cancel_listener(self):
        cancel = CancelListener()
        self.assertFalse(cancel.cancelled())
        cancel._on_sigint(None, None)
        self.assertTrue(cancel.cancelled())

    def test_vq2_gate1_fail_timeout_only(self):
        data = {"active_gate_index": 0, "race_status": {"race_finish_time_ns": -1}}
        self.assertFalse(gate1_fail(data, elapsed_s=5.0, pilot_gates_passed=0))
        self.assertTrue(gate1_fail(data, elapsed_s=16.0, pilot_gates_passed=0))

    def test_vq2_passed_first_gate_via_active_index(self):
        self.assertTrue(passed_first_gate({"active_gate_index": 1}))

    def test_vq2_course_complete_via_finish(self):
        data = {
            "race_status": {"race_finish_time_ns": 100},
            "active_gate_index": 3,
        }
        self.assertTrue(course_complete(data))

    def test_vq2_preflight_ready_without_track_gates(self):
        data = {
            "race_status": {"sim_boot_time_ms": 0},
            "camera": {"received_at": 1.0},
        }
        self.assertTrue(wait_for_session_ready(data, timeout_s=0.5))

    def test_race_go_already_passed_before_go(self):
        data = {
            "race_status": {
                "sim_boot_time_ms": 1000,
                "race_start_boot_time_ms": 5000,
            }
        }
        self.assertFalse(race_go_already_passed(data))

    def test_fresh_race_start_vq2_accepts_new_baseline(self):
        data = {
            "_preflight_race_start_baseline": 1000,
            "race_status": {
                "sim_boot_time_ms": 500,
                "race_start_boot_time_ms": 2000,
            },
        }
        self.assertTrue(wait_for_fresh_race_start_vq2(data, timeout_s=0.5))

    def test_run_auto_flight_preflight_fail_without_vision(self):
        controller = MagicMock()
        pilot = MagicMock()
        pilot.gates_passed = 0
        shared_data = {"race_status": {"sim_boot_time_ms": 0}}
        with patch("simulator.auto_flight.wait_for_session_ready", return_value=False):
            outcome, was_flying = run_auto_flight_loop(controller, pilot, shared_data)
        self.assertEqual(outcome, "preflight_fail")
        self.assertFalse(was_flying)

    @patch("simulator.auto_flight.CancelListener")
    def test_run_auto_flight_cancelled_during_preflight(self, mock_cl):
        cancel = MagicMock()
        cancel.cancelled.return_value = True
        mock_cl.return_value = cancel
        controller = MagicMock()
        pilot = MagicMock()
        shared_data = {}
        with patch("simulator.auto_flight.wait_for_session_ready", return_value=False):
            outcome, was_flying = run_auto_flight_loop(controller, pilot, shared_data)
        self.assertEqual(outcome, "cancelled")
        self.assertFalse(was_flying)

    @patch("simulator.auto_flight._run_attempt_preflight", return_value=True)
    @patch("simulator.auto_flight._retry_after_outcome", return_value=True)
    @patch("simulator.auto_flight.CancelListener")
    def test_run_auto_flight_retries_on_success(
        self, mock_cl, mock_retry, mock_preflight
    ):
        cancel = MagicMock()
        cancel.cancelled.return_value = False
        mock_cl.return_value = cancel

        controller = MagicMock()
        pilot = MagicMock()
        pilot.gates_passed = 0
        shared_data = {
            "gate_count": 1,
            "active_gate_index": 1,
            "race_status": {"race_finish_time_ns": 100},
        }

        with patch("simulator.auto_flight.course_complete", return_value=True):
            outcome, was_flying = run_auto_flight_loop(
                controller, pilot, shared_data
            )

        mock_retry.assert_called_once()
        self.assertEqual(mock_retry.call_args[0][0], "success")
        self.assertEqual(outcome, "cancelled")
        self.assertTrue(was_flying)


if __name__ == "__main__":
    unittest.main()
