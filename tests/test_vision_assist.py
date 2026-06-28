"""Tests for telemetry vision assist."""

import math
import unittest

from simulator.vision_assist import blend_telemetry_bearing


class TestVisionAssist(unittest.TestCase):
    def test_no_gate_target_returns_mavlink(self):
        err = 0.5
        self.assertAlmostEqual(blend_telemetry_bearing(err, None), err)

    def test_low_confidence_returns_mavlink(self):
        gt = {"bearing_rad": 0.3, "confidence": 0.2}
        err = 0.5
        self.assertAlmostEqual(blend_telemetry_bearing(err, gt), err)

    def test_blends_when_confident(self):
        gt = {
            "bearing_rad": 0.1,
            "confidence": 0.8,
            "_camera_received_at": 100.0,
        }
        err = 0.5
        out = blend_telemetry_bearing(err, gt, now=100.5)
        self.assertNotAlmostEqual(out, err, places=3)
        self.assertLess(abs(out), abs(err))


if __name__ == "__main__":
    unittest.main()
