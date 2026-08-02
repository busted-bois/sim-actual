"""Lock the live hover anchor and checkpoint-to-live action remap."""

import unittest

import numpy as np

from rl.deploy import live_scale_action
from rl.experts.fly2_course import HOVER_T


class HoverThrustInvariantTests(unittest.TestCase):
    def test_hover_thrust_anchor(self) -> None:
        self.assertEqual(HOVER_T, 0.27)

    def test_legacy_checkpoint_action_remap(self) -> None:
        action = np.array([0.75, -0.5, 0.4, 0.0])
        metadata = {"action_scale": [4.0, 4.0, 3.0], "train_hover": 0.5}
        expected = np.array([0.6, -0.6, 0.6, 0.27])
        np.testing.assert_array_equal(live_scale_action(action, metadata), expected)


if __name__ == "__main__":
    unittest.main()
