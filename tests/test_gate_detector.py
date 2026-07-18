"""Centroid gate detector (simulator/gate_detector.py)."""

import unittest

import cv2
import numpy as np

from simulator.gate_detector import _desaturated_orange_mask, detect_gate

# GATE_HEX_COLOR #F3390F as BGR.
_GATE_BGR = (15, 57, 243)
_BG = 30


def _frame(w=640, h=360):
    return np.full((h, w, 3), _BG, np.uint8)


def _draw_gate(img, cx, cy, outer_half, inner_half, color=_GATE_BGR):
    cv2.rectangle(
        img,
        (cx - outer_half, cy - outer_half),
        (cx + outer_half, cy + outer_half),
        color,
        -1,
    )
    cv2.rectangle(
        img,
        (cx - inner_half, cy - inner_half),
        (cx + inner_half, cy + inner_half),
        (_BG, _BG, _BG),
        -1,
    )


class DesaturatedMaskTests(unittest.TestCase):
    def test_blank_frame_not_fully_masked(self):
        """Regression: (1,1,3) inRange bounds used to match every pixel."""
        hsv = cv2.cvtColor(_frame(), cv2.COLOR_BGR2HSV)
        mask = _desaturated_orange_mask(hsv)
        self.assertLess(int(mask.sum() // 255), mask.size // 4)


class DetectGateTests(unittest.TestCase):
    def test_empty_frame_returns_none(self):
        self.assertIsNone(detect_gate(_frame(), frame_id=0, sim_time_ns=0))

    def test_detects_saturated_gate(self):
        img = _frame()
        _draw_gate(img, 320, 180, outer_half=70, inner_half=40)
        det = detect_gate(img, frame_id=1, sim_time_ns=1_000_000)
        self.assertIsNotNone(det)
        self.assertAlmostEqual(det.centroid_x_px, 320.0, delta=5.0)
        self.assertAlmostEqual(det.centroid_y_px, 180.0, delta=5.0)

    def test_desaturated_orange_fallback(self):
        img = _frame()
        _draw_gate(img, 320, 180, 70, 40, color=(90, 110, 160))
        self.assertIsNotNone(detect_gate(img, frame_id=2, sim_time_ns=2_000_000))


if __name__ == "__main__":
    unittest.main()
