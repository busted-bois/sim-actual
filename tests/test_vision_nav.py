import unittest

import numpy as np

from simulator.vision_nav import VisionGuidance

_QUAT_LEVEL = np.array([1.0, 0.0, 0.0, 0.0])
_DET = {"conf": 0.9, "pose": {"gate_pos_body": [8.0, 2.0, 0.0]}}


def _acquire(guide, n_frames=3, t0=0.0):
    """Feed the same detection for n_frames distinct frames until confirmed."""
    cmd = None
    for i in range(n_frames):
        cmd = guide.update(
            [_DET], np.zeros(3), np.zeros(3), _QUAT_LEVEL, 0.0, t0 + 0.05 * i
        )
    return cmd


class VisionGuidanceTests(unittest.TestCase):
    def test_min_hits_needed_before_chase(self):
        guide = VisionGuidance()
        cmd = guide.update([_DET], np.zeros(3), np.zeros(3), _QUAT_LEVEL, 0.0, 0.0)
        self.assertTrue(cmd.status.startswith("SCAN"))
        cmd = _acquire(guide)
        self.assertTrue(cmd.status.startswith("GO"))

    def test_yaw_tracks_target_bearing(self):
        guide = VisionGuidance()
        cmd = _acquire(guide)
        # Gate 2m right, 8m ahead -> yaw toward it (positive, ~atan2(2,8)).
        self.assertAlmostEqual(cmd.yaw_err, np.arctan2(2.0, 8.0), delta=1e-6)
        self.assertLessEqual(abs(cmd.yaw_err), guide.p["yaw_track_clip"])

    def test_brake_is_capped(self):
        guide = VisionGuidance()
        _acquire(guide)
        # Rush 2m toward the gate in 0.2s: closing-rate spike demands a hard
        # brake; backward tilt must clamp at max_brake, not max_tilt.
        pos = np.array([2.0, 0.0, 0.0])
        cmd = guide.update([_DET], pos, np.zeros(3), _QUAT_LEVEL, 0.0, 0.3)
        self.assertTrue(cmd.status.startswith("GO"))
        self.assertLessEqual(cmd.tgt_pitch, guide.p["max_brake"] + 1e-9)
        self.assertLess(guide.p["max_brake"], guide.p["max_tilt"])


if __name__ == "__main__":
    unittest.main()
