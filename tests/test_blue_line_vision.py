"""Unit tests for dual-cyan corridor detection."""

import unittest

import cv2
import numpy as np

from simulator.blue_line_vision import detect_blue_lines

# Bright cyan BGR → HSV ~H90-100, inside detector band.
_CYAN_BGR = (255, 255, 0)
_BG = 20


def _frame(w=640, h=360):
    return np.full((h, w, 3), _BG, np.uint8)


def _draw_corridor(img, left_x, right_x, *, y0=None, y1=None, thickness=8):
    h, w = img.shape[:2]
    if y0 is None:
        y0 = int(h * 0.55)
    if y1 is None:
        y1 = h - 5
    cv2.line(img, (left_x, y1), (left_x, y0), _CYAN_BGR, thickness)
    cv2.line(img, (right_x, y1), (right_x, y0), _CYAN_BGR, thickness)


class DetectBlueLinesTests(unittest.TestCase):
    def test_empty_frame_not_found(self):
        est, _ = detect_blue_lines(_frame(), frame_id=0)
        self.assertFalse(est.found)

    def test_centered_corridor(self):
        img = _frame()
        _draw_corridor(img, 200, 440)
        est, mask = detect_blue_lines(img, frame_id=1, return_mask=True)
        self.assertTrue(est.found)
        self.assertTrue(est.left_found)
        self.assertTrue(est.right_found)
        self.assertAlmostEqual(est.cx_norm, 0.0, delta=0.15)
        self.assertGreater(est.width_norm, 0.2)
        self.assertIsNotNone(mask)

    def test_corridor_shifted_right(self):
        img = _frame()
        _draw_corridor(img, 280, 520)
        est, _ = detect_blue_lines(img, frame_id=2)
        self.assertTrue(est.found)
        self.assertGreater(est.cx_norm, 0.1)

    def test_corridor_shifted_left(self):
        img = _frame()
        _draw_corridor(img, 80, 320)
        est, _ = detect_blue_lines(img, frame_id=3)
        self.assertTrue(est.found)
        self.assertLess(est.cx_norm, -0.1)

    def test_ignores_orange_gate(self):
        img = _frame()
        # Orange gate-like blob should not register as blue lines.
        cv2.rectangle(img, (250, 80), (390, 220), (15, 57, 243), -1)
        est, _ = detect_blue_lines(img, frame_id=4)
        self.assertFalse(est.found)

    def test_heading_when_far_mid_offset(self):
        img = _frame()
        h, w = img.shape[:2]
        # Near: centered; far: vanishes to the right.
        cv2.line(img, (200, h - 5), (200, int(h * 0.55)), _CYAN_BGR, 8)
        cv2.line(img, (440, h - 5), (440, int(h * 0.55)), _CYAN_BGR, 8)
        cv2.line(img, (260, int(h * 0.5)), (300, 20), _CYAN_BGR, 8)
        cv2.line(img, (500, int(h * 0.5)), (540, 20), _CYAN_BGR, 8)
        est, _ = detect_blue_lines(img, frame_id=5)
        self.assertTrue(est.found)
        self.assertGreater(est.heading_err, 0.0)


if __name__ == "__main__":
    unittest.main()
