"""CV inner-corner extractor (simulator/gate_corners_cv.py)."""

import unittest

import cv2
import numpy as np

from simulator.gate_corners_cv import find_gate_inner_corners, order_corners

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


class OrderCornersTests(unittest.TestCase):
    def test_orders_shuffled_square_tl_tr_br_bl(self):
        quad = np.array([[10, 10], [50, 10], [50, 50], [10, 50]], float)
        rng = np.random.default_rng(1)
        for _ in range(10):
            shuffled = quad[rng.permutation(4)]
            np.testing.assert_array_equal(order_corners(shuffled), quad)


class FindGateInnerCornersTests(unittest.TestCase):
    def test_recovers_inner_corners(self):
        img = _frame()
        cx, cy, ih = 320, 180, 40
        _draw_gate(img, cx, cy, outer_half=70, inner_half=ih)
        corners = find_gate_inner_corners(img)
        self.assertIsNotNone(corners)
        expected = np.array(
            [
                [cx - ih, cy - ih],
                [cx + ih, cy - ih],
                [cx + ih, cy + ih],
                [cx - ih, cy + ih],
            ],
            float,
        )
        np.testing.assert_allclose(corners, expected, atol=3.0)

    def test_ring_without_resolvable_hole_shrinks_outer(self):
        # Hole too small for the mask to keep -> falls back to shrinking the
        # outer quad by the 1.5/2.7 inner/outer ratio.
        img = _frame()
        cx, cy, oh = 320, 180, 54
        cv2.rectangle(img, (cx - oh, cy - oh), (cx + oh, cy + oh), _GATE_BGR, -1)
        corners = find_gate_inner_corners(img)
        self.assertIsNotNone(corners)
        ih = oh * (1.5 / 2.7)
        expected = np.array(
            [
                [cx - ih, cy - ih],
                [cx + ih, cy - ih],
                [cx + ih, cy + ih],
                [cx - ih, cy + ih],
            ],
            float,
        )
        np.testing.assert_allclose(corners, expected, atol=3.0)

    def test_desaturated_orange_fallback(self):
        # VQ2 R2 scanned-gate look: same hue family, low saturation.
        img = _frame()
        _draw_gate(img, 320, 180, 70, 40, color=(90, 110, 160))
        self.assertIsNotNone(find_gate_inner_corners(img))

    def test_too_small_gate_rejected(self):
        img = _frame()
        _draw_gate(img, 320, 180, outer_half=5, inner_half=2)
        self.assertIsNone(find_gate_inner_corners(img))

    def test_edge_clipped_gate_rejected(self):
        # Opening partially out of frame -> corners pinned to the border are
        # clip artifacts; PnP on them would solve a mirrored pose.
        img = _frame()
        _draw_gate(img, 10, 180, outer_half=40, inner_half=20)
        self.assertIsNone(find_gate_inner_corners(img))

    def test_empty_frame(self):
        self.assertIsNone(find_gate_inner_corners(_frame()))

    def test_bad_input(self):
        self.assertIsNone(find_gate_inner_corners(None))
        self.assertIsNone(find_gate_inner_corners(np.zeros((360, 640), np.uint8)))


if __name__ == "__main__":
    unittest.main()
