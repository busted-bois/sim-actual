import unittest
from unittest.mock import MagicMock

from rl.fly2_course import (
    Fly2Config,
    Fly2CoursePilot,
    compute_course_rates,
    detect_climb_course,
    resolve_gate_map,
    track_gates_to_gate_map,
)


class Fly2CourseTests(unittest.TestCase):
    def test_track_gates_to_gate_map(self):
        track = [
            {
                "gate_id": 0,
                "position_ned": (1.0, 2.0, 3.0),
                "orientation_ned": (1.0, 0.0, 0.0, 0.0),
                "width": 2.7,
                "height": 2.7,
            }
        ]
        gm = track_gates_to_gate_map(track)
        self.assertEqual(len(gm), 1)
        self.assertEqual(gm[0]["pos"], [1.0, 2.0, 3.0])
        self.assertEqual(gm[0]["id"], 0)

    def test_compute_course_rates_returns_four(self):
        gate_map = [{"pos": [10.0, 0.0, 0.0]}]
        cfg = Fly2Config()
        out = compute_course_rates(
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0, 0.0),
            0,
            gate_map,
            0.0,
            cfg,
        )
        self.assertEqual(len(out), 4)

    def test_gates_passed_always_zero_for_fly2(self):
        controller = MagicMock()
        pilot = Fly2CoursePilot(controller, {"active_gate_index": 2})
        self.assertEqual(pilot.gates_passed, 0)

    def test_detect_climb_course(self):
        flat = [{"pos": [0.0, 0.0, 0.0]}, {"pos": [0.0, 0.0, 1.0]}]
        climb = [{"pos": [0.0, 0.0, 0.0]}, {"pos": [0.0, 0.0, 5.0]}]
        self.assertFalse(detect_climb_course(flat))
        self.assertTrue(detect_climb_course(climb))

    def test_fly2_config_main_defaults(self):
        cfg = Fly2Config()
        self.assertEqual(cfg.speed, 2.8)
        self.assertEqual(cfg.klat, 0.04)

    def test_resolve_gate_map_uses_json_when_vq2_nulled(self):
        data = {
            "track_positions_valid": False,
            "track_gates": [
                {
                    "gate_id": 0,
                    "position_ned": (0.0, 0.0, 0.0),
                    "orientation_ned": (1.0, 0.0, 0.0, 0.0),
                }
            ],
        }
        gm = resolve_gate_map(data)
        self.assertGreater(len(gm), 0)

    def test_on_attempt_start_sets_flipz_for_climb(self):
        controller = MagicMock()
        track = [
            {
                "position_ned": (1.0, 0.0, 0.0),
                "orientation_ned": (1.0, 0.0, 0.0, 0.0),
            },
            {
                "position_ned": (0.0, 0.0, 5.0),
                "orientation_ned": (1.0, 0.0, 0.0, 0.0),
            },
        ]
        pilot = Fly2CoursePilot(controller, {"track_gates": track})
        pilot.on_attempt_start()
        self.assertTrue(pilot.config.flipz)

    def test_reset_for_attempt_clears_stale_gate_target(self):
        controller = MagicMock()
        data = {"gate_target": {"detected": True, "frame_id": 7}}
        pilot = Fly2CoursePilot(controller, data)
        pilot.reset_for_attempt()
        self.assertNotIn("gate_target", data)


if __name__ == "__main__":
    unittest.main()
