"""Unit tests for dual-cyan corridor detection."""

import math
import unittest

import cv2
import numpy as np

from simulator.blue_line_vision import (
    _MAX_CX_STEP,
    _MIN_HDG_SPAN_FRAC,
    BlueLineTracker,
    _fit,
    cyan_mask,
    detect_blue_lines,
    estimate_from_dict,
    estimate_to_dict,
)

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


def _draw_bend(img, near_l, far_l, near_r, far_r, *, thickness=10):
    h = img.shape[0]
    cv2.line(img, (near_l, h - 5), (far_l, 60), _CYAN_BGR, thickness)
    cv2.line(img, (near_r, h - 5), (far_r, 60), _CYAN_BGR, thickness)


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
        self.assertEqual(mask.shape[:2], img.shape[:2])

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


class OffCentreCorridorTests(unittest.TestCase):
    """The fixed w//2 split used to INVERT cx once both rails shared a half."""

    def test_corridor_wholly_in_right_half(self):
        img = _frame()
        _draw_corridor(img, 360, 560, y0=150)
        est, _ = detect_blue_lines(img, frame_id=0)
        self.assertTrue(est.found)
        # True corridor mid is 460 px of 640 -> cx +0.44 (old detector: -0.06).
        self.assertGreater(est.cx_norm, 0.3)
        self.assertTrue(est.left_found and est.right_found)
        self.assertFalse(est.single_rail)

    def test_corridor_wholly_in_left_half(self):
        img = _frame()
        _draw_corridor(img, 80, 280, y0=150)
        est, _ = detect_blue_lines(img, frame_id=0)
        self.assertTrue(est.found)
        self.assertLess(est.cx_norm, -0.3)


class HeadingSignTests(unittest.TestCase):
    """A hard bend puts both far rails in one half; heading used to invert."""

    def test_hard_right_bend(self):
        img = _frame()
        _draw_bend(img, 200, 430, 440, 640)
        est, _ = detect_blue_lines(img, frame_id=0)
        self.assertTrue(est.found)
        self.assertGreater(est.heading_err, 0.3)  # old detector: -0.02 rad

    def test_hard_left_bend(self):
        img = _frame()
        _draw_bend(img, 200, -30, 440, 180)
        est, _ = detect_blue_lines(img, frame_id=0)
        self.assertTrue(est.found)
        self.assertLess(est.heading_err, -0.3)  # old detector: +0.03 rad

    def test_straight_corridor_has_no_heading(self):
        img = _frame()
        _draw_bend(img, 200, 200, 440, 440)
        est, _ = detect_blue_lines(img, frame_id=0)
        self.assertTrue(est.found)
        self.assertLess(abs(est.heading_err), 0.05)


class MaskTests(unittest.TestCase):
    def test_thin_far_ribbon_survives(self):
        """MORPH_OPEN 5x5 wiped 2 px far rails (830 px -> 0), killing heading."""
        img = _frame()
        cv2.line(img, (300, 200), (310, 60), _CYAN_BGR, 2)
        cv2.line(img, (360, 200), (350, 60), _CYAN_BGR, 2)
        self.assertGreater(cv2.countNonZero(cyan_mask(img)), 200)

    def test_blue_sky_excluded(self):
        """Sky sits at H~113; the old cyan band reached H=140 and ate it."""
        img = _frame()
        img[0:120, :] = (200, 80, 40)  # blue-ish sky, BGR
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        self.assertGreater(int(hsv[10, 10, 0]), 106)  # guard the fixture
        self.assertEqual(cv2.countNonZero(cyan_mask(img)[0:120, :]), 0)

    def test_sky_does_not_corrupt_heading(self):
        img = _frame()
        img[0:120, :] = (200, 80, 40)
        _draw_corridor(img, 200, 440, y0=220)
        est, _ = detect_blue_lines(img, frame_id=0)
        self.assertTrue(est.found)
        self.assertLess(abs(est.heading_err), 0.15)


class SingleRailTests(unittest.TestCase):
    def test_single_rail_centre_is_offset_not_centred(self):
        img = _frame()
        cv2.line(img, (150, 355), (150, 120), _CYAN_BGR, 10)
        est, _ = detect_blue_lines(img, frame_id=0)
        self.assertTrue(est.found)
        self.assertTrue(est.single_rail)
        # A lone LEFT rail means the corridor lies to its right.
        self.assertGreater(est.cx_norm, (150 - 320) / 320.0)

    def test_tracker_carries_width_into_single_rail_frame(self):
        tracker = BlueLineTracker()
        paired = _frame()
        _draw_corridor(paired, 150, 450, y0=120)
        est, _ = tracker.update(paired, 1)
        self.assertFalse(est.single_rail)
        remembered = est.width_norm

        lone = _frame()
        cv2.line(lone, (150, 355), (150, 120), _CYAN_BGR, 10)
        est2, _ = tracker.update(lone, 2)
        self.assertTrue(est2.single_rail)
        # Centre reconstructed one half-width right of the rail it can see.
        expect = (150 + 0.5 * remembered * 640 - 320) / 320.0
        self.assertAlmostEqual(est2.cx_norm, expect, delta=0.12)

    def test_tracker_reset_clears_memory(self):
        tracker = BlueLineTracker()
        img = _frame()
        _draw_corridor(img, 150, 450, y0=120)
        tracker.update(img, 1)
        tracker.reset()
        self.assertIsNone(tracker._half_width_px)


class TrackerHeadingGuardTests(unittest.TestCase):
    def test_heading_step_is_rate_limited(self):
        tracker = BlueLineTracker()
        straight = _frame()
        _draw_bend(straight, 200, 200, 440, 440)
        est1, _ = tracker.update(straight, 1)
        self.assertLess(abs(math.degrees(est1.heading_err)), 5.0)

        hard = _frame()
        _draw_bend(hard, 200, 430, 440, 640)
        est2, _ = tracker.update(hard, 2)
        # Real bend is ~36 deg; one frame may not move more than the guard.
        self.assertLessEqual(abs(math.degrees(est2.heading_err)), 25.0 + 5.0)
        self.assertGreater(est2.heading_err, 0.0)
        self.assertLess(est2.conf, est1.conf)  # flagged as suspect

        est3, _ = tracker.update(hard, 3)
        self.assertGreater(est3.heading_err, est2.heading_err)  # converges


class HeadingValidityTests(unittest.TestCase):
    """heading_err is only published when the fit can actually support it."""

    def test_short_span_reports_no_heading(self):
        # Corridor confined to the bottom of the frame: the band centres span
        # too few rows to constrain a slope, so the honest answer is "none".
        img = _frame()
        _draw_corridor(img, 260, 380, y0=320, y1=355)
        est, _ = detect_blue_lines(img)
        if est.found:
            self.assertLess(est.span_frac, _MIN_HDG_SPAN_FRAC)
            self.assertFalse(est.heading_valid)
            self.assertEqual(est.heading_err, 0.0)

    def test_full_height_bend_reports_a_heading(self):
        img = _frame()
        _draw_bend(img, 200, 430, 440, 640)
        est, _ = detect_blue_lines(img)
        self.assertTrue(est.found)
        self.assertGreaterEqual(est.span_frac, _MIN_HDG_SPAN_FRAC)
        self.assertTrue(est.heading_valid)
        self.assertGreater(est.heading_err, 0.0)  # bends right

    def test_span_frac_is_reported_even_when_gated(self):
        img = _frame()
        _draw_corridor(img, 200, 440)
        est, _ = detect_blue_lines(img)
        self.assertTrue(est.found)
        self.assertGreater(est.span_frac, 0.0)
        self.assertGreaterEqual(est.paired_bands, 1)


class RailFitTests(unittest.TestCase):
    def test_fitted_rails_cannot_cross(self):
        # Independently fitted rails invert on 15.4% of real frames, which
        # flips the inferred side and puts the centre a full width wrong.
        # centre +- |half_width| makes that unrepresentable.
        vs = [40.0, 90.0, 140.0]
        pl = [100.0, 90.0, 80.0]
        pr = [110.0, 140.0, 170.0]  # diverging fast enough to cross above
        fit_c = _fit(vs, [0.5 * (a + b) for a, b in zip(pl, pr)])
        fit_hw = _fit(vs, [0.5 * (b - a) for a, b in zip(pl, pr)])
        indep_l, indep_r = _fit(vs, pl), _fit(vs, pr)
        crossed = False
        for v in range(-200, 400, 5):
            lo = fit_c(v) - abs(fit_hw(v))
            hi = fit_c(v) + abs(fit_hw(v))
            self.assertLessEqual(lo, hi, f"centre-fit rails crossed at v={v}")
            crossed = crossed or indep_r(v) < indep_l(v)
        self.assertTrue(crossed, "test data must actually cross under indep fits")


class CxFilterTests(unittest.TestCase):
    def test_single_frame_jump_is_clamped(self):
        # cx reaches the roll command through TRACK_LAT_GAIN and had no filter
        # at all; measured jumps up to 0.709 in one frame.
        tracker = BlueLineTracker()
        left = _frame()
        _draw_corridor(left, 120, 260)
        est1, _ = tracker.update(left, 1)
        self.assertTrue(est1.found)

        right = _frame()
        _draw_corridor(right, 400, 540)
        est2, _ = tracker.update(right, 2)
        self.assertTrue(est2.found)
        self.assertLessEqual(abs(est2.cx_norm - est1.cx_norm), _MAX_CX_STEP + 1e-6)
        self.assertLess(est2.conf, est1.conf)  # flagged as suspect

        est3, _ = tracker.update(right, 3)
        self.assertGreater(est3.cx_norm, est2.cx_norm)  # converges toward truth

    def test_reset_clears_cx_memory(self):
        tracker = BlueLineTracker()
        left = _frame()
        _draw_corridor(left, 120, 260)
        tracker.update(left, 1)
        tracker.reset()
        right = _frame()
        _draw_corridor(right, 400, 540)
        est, _ = tracker.update(right, 2)
        raw, _ = detect_blue_lines(right)
        self.assertAlmostEqual(est.cx_norm, raw.cx_norm, places=6)


class SerializationTests(unittest.TestCase):
    def test_round_trip_preserves_new_fields(self):
        img = _frame()
        _draw_corridor(img, 200, 440)
        est, _ = detect_blue_lines(img, frame_id=7)
        back = estimate_from_dict(estimate_to_dict(est))
        self.assertEqual(back.found, est.found)
        self.assertAlmostEqual(back.cx_norm, est.cx_norm)
        self.assertAlmostEqual(back.conf, est.conf)
        self.assertEqual(back.single_rail, est.single_rail)
        self.assertEqual(back.frame_id, 7)
        self.assertEqual(len(back.points), len(est.points))
        self.assertEqual(back.heading_valid, est.heading_valid)
        self.assertAlmostEqual(back.span_frac, est.span_frac)
        self.assertEqual(back.paired_bands, est.paired_bands)


if __name__ == "__main__":
    unittest.main()
