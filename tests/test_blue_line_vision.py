"""Unit tests for dual-cyan corridor detection."""

import os
import unittest
from unittest import mock

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


def _draw_ribbons(img, left_inner, right_inner, *, left_w=8, right_w=8):
    """Ribbons of independent width, growing AWAY from the corridor.

    Inner edges land exactly on left_inner/right_inner regardless of width,
    which is the whole point of the inner-edge estimator.
    """
    h = img.shape[0]
    y0, y1 = int(h * 0.55), h - 5
    cv2.rectangle(img, (left_inner - left_w, y0), (left_inner, y1), _CYAN_BGR, -1)
    cv2.rectangle(img, (right_inner, y0), (right_inner + right_w, y1), _CYAN_BGR, -1)


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


class InnerEdgeTests(unittest.TestCase):
    """Cases where inner edges beat the centroid midpoint."""

    def test_uses_edge_estimator_by_default(self):
        img = _frame()
        _draw_corridor(img, 200, 440)
        est, _ = detect_blue_lines(img, frame_id=10)
        self.assertEqual(est.source, "edge")
        self.assertGreaterEqual(est.edge_rows, 4)

    @mock.patch.dict(os.environ, {"BL_INNER_EDGE": "0"})
    def test_env_toggle_reverts_to_centroid(self):
        img = _frame()
        _draw_corridor(img, 200, 440)
        est, _ = detect_blue_lines(img, frame_id=11)
        self.assertEqual(est.source, "centroid")
        self.assertAlmostEqual(est.cx_norm, est.cx_centroid, delta=1e-9)

    def test_unequal_ribbon_width_does_not_bias_center(self):
        """Bloom on one ribbon drags its centroid — but not its inner edge."""
        img = _frame()
        # Corridor is centered (inner edges 200/440, mid 320 = frame center);
        # the left ribbon is 10x thicker, as if closer/brighter.
        _draw_ribbons(img, 200, 440, left_w=80, right_w=8)
        est, _ = detect_blue_lines(img, frame_id=12)
        self.assertTrue(est.found)
        self.assertEqual(est.source, "edge")
        # Inner edges see a centered corridor...
        self.assertAlmostEqual(est.cx_norm, 0.0, delta=0.03)
        # ...while the centroid midpoint is pulled toward the fat ribbon.
        self.assertLess(est.cx_centroid, -0.03)

    def test_both_ribbons_right_of_frame_center(self):
        """The w//2 split breaks when the corridor is off to one side."""
        img = _frame()
        _draw_ribbons(img, 360, 600)  # corridor mid 480 → cx = +0.5
        est, _ = detect_blue_lines(img, frame_id=13)
        self.assertTrue(est.found)
        self.assertEqual(est.source, "edge")
        self.assertAlmostEqual(est.cx_norm, 0.5, delta=0.06)
        # Centroid path lumps both ribbons into its "right" half and reports a
        # near-centered corridor — the failure this replaces.
        self.assertLess(abs(est.cx_centroid), 0.2)

    def test_falls_back_to_centroid_when_scan_cannot_track(self):
        """One ribbon = no corridor gap; the legacy estimate must still fly."""
        img = _frame()
        h = img.shape[0]
        cv2.line(img, (200, h - 5), (200, int(h * 0.55)), _CYAN_BGR, 8)
        est, _ = detect_blue_lines(img, frame_id=14)
        self.assertTrue(est.found)
        self.assertEqual(est.source, "centroid")
        self.assertTrue(est.left_found)
        self.assertFalse(est.right_found)

    def test_detached_reflection_inside_corridor_is_rejected(self):
        """A blob floating in the corridor must not become an inner edge."""
        img = _frame()
        h = img.shape[0]
        _draw_ribbons(img, 200, 440)
        # Reflection-like patch well inside the corridor, near the bottom.
        cv2.rectangle(img, (300, h - 40), (340, h - 20), _CYAN_BGR, -1)
        est, _ = detect_blue_lines(img, frame_id=15)
        self.assertTrue(est.found)
        self.assertAlmostEqual(est.cx_norm, 0.0, delta=0.06)


if __name__ == "__main__":
    unittest.main()
