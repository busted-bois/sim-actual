import unittest

from simulator.race_monitor import (
    course_complete,
    gate1_fail,
    gate1_watch_line,
    gate_count,
    gate_progress_stall,
    gate_progress_watch_line,
    passed_first_gate,
    passed_gate0_plane,
    signed_dist_gate0,
)


def _gate0_data(active=0, odometry=None):
    return {
        "active_gate_index": active,
        "track_gates": [
            {
                "position_ned": (0.0, 0.0, 0.0),
                "orientation_ned": (1.0, 0.0, 0.0, 0.0),
                "width": 2.72,
                "height": 2.72,
            }
        ],
        "odometry": odometry,
    }


class RaceMonitorTests(unittest.TestCase):
    def test_gate_count_empty(self):
        self.assertEqual(gate_count({}), 0)

    def test_gate_count_from_track(self):
        self.assertEqual(gate_count({"track_gates": [{}, {}, {}]}), 3)

    def test_passed_first_gate_at_zero(self):
        self.assertFalse(passed_first_gate({"active_gate_index": 0}))

    def test_passed_first_gate_at_one(self):
        self.assertTrue(passed_first_gate({"active_gate_index": 1}))

    def test_course_complete_via_finish_ns(self):
        data = {
            "track_gates": [{}, {}],
            "race_status": {"race_finish_time_ns": 0},
            "active_gate_index": 1,
        }
        self.assertTrue(course_complete(data))

    def test_course_complete_via_active_index(self):
        data = {
            "track_gates": [{}, {}],
            "race_status": {"race_finish_time_ns": -1},
            "active_gate_index": 2,
        }
        self.assertTrue(course_complete(data))

    def test_course_not_complete_mid_race(self):
        data = {
            "track_gates": [{}, {}, {}],
            "race_status": {"race_finish_time_ns": -1},
            "active_gate_index": 1,
        }
        self.assertFalse(course_complete(data))

    def test_course_not_complete_no_gates(self):
        self.assertFalse(course_complete({"active_gate_index": 99}))

    def test_gate1_fail_on_timeout(self):
        data = {"active_gate_index": 0}
        self.assertTrue(gate1_fail(data, elapsed_s=11.0, pilot_gates_passed=0))

    def test_gate1_fail_not_yet(self):
        data = {"active_gate_index": 0}
        self.assertFalse(gate1_fail(data, elapsed_s=10.0, pilot_gates_passed=0))

    def test_gate1_fail_early_past_plane(self):
        data = _gate0_data(
            odometry={"x": 5.0, "y": 0.0, "z": 0.0},
        )
        self.assertTrue(gate1_fail(data, elapsed_s=9.0, pilot_gates_passed=0))

    def test_gate1_no_fail_if_sim_passed(self):
        data = {"active_gate_index": 1}
        self.assertFalse(gate1_fail(data, elapsed_s=50.0, pilot_gates_passed=0))

    def test_gate1_no_fail_if_pilot_passed(self):
        data = _gate0_data(
            odometry={"x": 5.0, "y": 0.0, "z": 0.0},
        )
        self.assertFalse(gate1_fail(data, elapsed_s=20.0, pilot_gates_passed=1))

    def test_signed_dist_gate0(self):
        data = _gate0_data(odometry={"x": 3.0, "y": 0.0, "z": 0.0})
        signed = signed_dist_gate0(data)
        self.assertIsNotNone(signed)
        assert signed is not None
        self.assertAlmostEqual(signed, 3.0)

    def test_passed_gate0_plane(self):
        data = _gate0_data(odometry={"x": 1.0, "y": 0.0, "z": 0.0})
        self.assertTrue(passed_gate0_plane(data))

    def test_gate1_watch_line(self):
        line = gate1_watch_line({"active_gate_index": 0}, 12.0, 0)
        self.assertIn("gate1_watch", line)
        self.assertIn("active=0", line)
        self.assertIn("elapsed=12s", line)

    def test_gate_progress_stall_not_yet(self):
        data = {
            "track_gates": [{}, {}, {}],
            "race_status": {"race_finish_time_ns": -1},
            "active_gate_index": 1,
        }
        self.assertFalse(gate_progress_stall(data, last_active=1, elapsed_since_advance_s=19.0))

    def test_gate_progress_stall_on_timeout(self):
        data = {
            "track_gates": [{}, {}, {}],
            "race_status": {"race_finish_time_ns": -1},
            "active_gate_index": 1,
        }
        self.assertTrue(gate_progress_stall(data, last_active=1, elapsed_since_advance_s=20.0))

    def test_gate_progress_stall_false_when_complete(self):
        data = {
            "track_gates": [{}, {}],
            "race_status": {"race_finish_time_ns": 0},
            "active_gate_index": 1,
        }
        self.assertFalse(gate_progress_stall(data, last_active=1, elapsed_since_advance_s=30.0))

    def test_gate_progress_watch_line(self):
        line = gate_progress_watch_line({"active_gate_index": 2}, 1, 15.0)
        self.assertIn("gate_progress_watch", line)
        self.assertIn("active=2", line)
        self.assertIn("last_active=1", line)


if __name__ == "__main__":
    unittest.main()
