import math
import unittest
from unittest.mock import MagicMock

from rl.experts.fly2_course import HOVER_T
from simulator.vision_nav import Cmd
from simulator.vision_nav_pilot import VisionNavPilot

_LEVEL_ODO = {
    "x": 0.0,
    "y": 0.0,
    "z": -5.0,
    "vx": 0.0,
    "vy": 0.0,
    "vz": 0.0,
    "qw": 1.0,
    "qx": 0.0,
    "qy": 0.0,
    "qz": 0.0,
}

# 180-degree roll: upside down.
_FLIPPED_ODO = dict(_LEVEL_ODO, qw=0.0, qx=1.0)


class VisionNavPilotTests(unittest.TestCase):
    def _make_pilot(self, data):
        controller = MagicMock()
        pilot = VisionNavPilot(controller, data)
        controller.reset_mock()
        return pilot, controller

    def test_gates_passed_tracks_guidance(self):
        pilot, _ = self._make_pilot({})
        self.assertEqual(pilot.gates_passed, 0)
        pilot.guide.passed.append(None)
        self.assertEqual(pilot.gates_passed, 1)

    def test_tick_hovers_without_pose(self):
        pilot, controller = self._make_pilot({})
        pilot.tick()
        controller.set_attitude_rates.assert_called_once_with(0, 0, 0, HOVER_T)

    def test_tick_converts_guidance_cmd_to_rates(self):
        data = {"odometry": dict(_LEVEL_ODO)}
        pilot, controller = self._make_pilot(data)
        pilot.guide = MagicMock()
        pilot.guide.update.return_value = Cmd(0.0, -0.1, 0.0, -5.0, "GO")
        pilot.guide.n_passed = 0
        pilot.tick()
        args = controller.set_attitude_rates.call_args[0]
        roll_cmd, pitch_cmd, yaw_cmd, thrust = args
        self.assertEqual(roll_cmd, 0.0)
        # tgt_pitch=-0.1 from level: SIGN_PITCH * K_ATT * (-0.1 - 0) = -0.06
        self.assertAlmostEqual(pitch_cmd, -0.06, places=6)
        self.assertEqual(yaw_cmd, 0.0)
        # Level at target altitude -> hover thrust.
        self.assertAlmostEqual(thrust, HOVER_T, places=6)

    def test_tick_passes_pose_gates_to_guidance_once_per_frame(self):
        gates = [{"conf": 0.9, "pose": {"gate_pos_body": [5.0, 0.0, 0.0]}}]
        data = {
            "odometry": dict(_LEVEL_ODO),
            "pose": {"frame_id": 1, "gates": gates},
        }
        pilot, _ = self._make_pilot(data)
        pilot.guide = MagicMock()
        pilot.guide.update.return_value = Cmd(0.0, 0.0, 0.0, -5.0, "GO")
        pilot.guide.n_passed = 0
        pilot.tick()
        self.assertIs(pilot.guide.update.call_args[0][0], gates)
        # Same inference frame on the next tick: detections must NOT be fed
        # again (min_hits counts frames, not 250 Hz ticks).
        pilot.tick()
        self.assertEqual(pilot.guide.update.call_args[0][0], [])
        # New frame feeds again.
        data["pose"] = {"frame_id": 2, "gates": gates}
        pilot.tick()
        self.assertIs(pilot.guide.update.call_args[0][0], gates)

    def test_unsafe_flipped_cuts_to_hover_after_five_ticks(self):
        data = {"odometry": dict(_FLIPPED_ODO)}
        pilot, controller = self._make_pilot(data)
        pilot.guide = MagicMock()
        pilot.guide.update.return_value = Cmd(0.0, 0.0, 0.0, -5.0, "GO")
        pilot.guide.n_passed = 0
        for _ in range(5):
            pilot.tick()
        self.assertEqual(controller.set_attitude_rates.call_args[0], (0, 0, 0, HOVER_T))

    def test_on_attempt_start_no_flipz_for_ned_track_burst(self):
        # Climb course in NED: higher gates have MORE NEGATIVE z, so the
        # up-positive climb heuristic must not fire on a live track burst.
        track = [
            {"position_ned": (10.0, 0.0, -2.0), "orientation_ned": (1, 0, 0, 0)},
            {"position_ned": (0.0, 0.0, -8.0), "orientation_ned": (1, 0, 0, 0)},
        ]
        data = {"track_gates": track}
        pilot, _ = self._make_pilot(data)
        pilot.on_attempt_start()
        self.assertFalse(pilot._pose._flipz)

    def test_reset_for_attempt_clears_stale_vision_state(self):
        data = {
            "gate_target": {"detected": True},
            "pose": {"gates": []},
        }
        pilot, controller = self._make_pilot(data)
        pilot.guide.passed.append(None)
        pilot.reset_for_attempt()
        self.assertNotIn("gate_target", data)
        self.assertNotIn("pose", data)
        self.assertEqual(pilot.gates_passed, 0)
        controller.set_attitude_rates.assert_called_with(0, 0, 0, HOVER_T)

    def test_status_string_is_finite(self):
        # Guidance with no detections should SCAN, producing sane rate cmds.
        data = {"odometry": dict(_LEVEL_ODO)}
        pilot, controller = self._make_pilot(data)
        pilot.tick()
        args = controller.set_attitude_rates.call_args[0]
        self.assertTrue(all(math.isfinite(a) for a in args))


if __name__ == "__main__":
    unittest.main()
