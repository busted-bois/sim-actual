"""Tests for gate-orientation context in the camera-native fallback."""

import unittest

import numpy as np

from rl.experts.fly2_course import HOVER_T
from rl.experts.vision_fallback import FallbackBrain
from simulator.gyro_ahrs import euler_to_quat


def _level_quat() -> np.ndarray:
    return np.array(euler_to_quat(0.0, 0.0, 0.0), dtype=np.float64)


class FallbackBrainGateContextTests(unittest.TestCase):
    def test_set_and_reset(self):
        fallback = FallbackBrain()
        self.assertIsNone(fallback._gate_quat)
        fallback.set_gate_context(_level_quat())
        np.testing.assert_array_equal(fallback._gate_quat, _level_quat())
        fallback.reset()
        self.assertIsNone(fallback._gate_quat)

    def test_none_safe(self):
        fallback = FallbackBrain()
        fallback.set_gate_context(None)
        command = fallback.update({}, _level_quat(), 1.0 / 60.0)
        self.assertAlmostEqual(command[3], HOVER_T)


if __name__ == "__main__":
    unittest.main()
