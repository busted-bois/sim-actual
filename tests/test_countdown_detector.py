import unittest

import numpy as np

from simulator.countdown_detector import (
    countdown_gate_cleared,
    countdown_visible,
    reset_countdown_gate,
    update_countdown_gate,
)


class CountdownDetectorTests(unittest.TestCase):
    def test_countdown_visible_on_bright_text_roi(self):
        img = np.zeros((240, 320, 3), dtype=np.uint8)
        img[20:80, 120:200] = 255
        self.assertTrue(countdown_visible(img))

    def test_countdown_not_visible_on_blank(self):
        img = np.zeros((240, 320, 3), dtype=np.uint8)
        self.assertFalse(countdown_visible(img))

    def test_gate_cleared_after_frames(self):
        data = {}
        reset_countdown_gate(data)
        update_countdown_gate(data, True)
        self.assertFalse(countdown_gate_cleared(data))
        for _ in range(5):
            update_countdown_gate(data, False)
        self.assertTrue(countdown_gate_cleared(data))


if __name__ == "__main__":
    unittest.main()
