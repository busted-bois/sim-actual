"""Perception synthetic QA: deterministic scene, UNet forward, PnP recovery, and
graceful missing-weights behavior (GateNet weights are absent in this repo)."""

import os
import unittest
import warnings

import numpy as np
import torch

from rl.perception.gatenet import TRAIN_H, TRAIN_W, WEIGHTS_PATH
from rl.perception.gatenet import UNet
from rl.perception.pnp import estimate_pose
from rl.perception.synthetic import make_synthetic_gate
from rl.perception.vision_fusion import gate_pose_from_image

# Synthetic corners are exact projections, so PnP recovers near-exactly. A loose
# bound keeps the test robust to float rounding without hiding real breakage.
RANGE_TOL_M = 0.05


class SyntheticSceneTests(unittest.TestCase):
    def test_shapes_dtypes_and_finite(self):
        img, corners = make_synthetic_gate(seed=0)
        self.assertEqual(tuple(img.shape), (1, 3, TRAIN_H, TRAIN_W))
        self.assertEqual(img.dtype, torch.float32)
        self.assertTrue(bool(torch.isfinite(img).all()))
        self.assertEqual(corners.shape, (4, 2))
        self.assertTrue(np.issubdtype(corners.dtype, np.floating))
        self.assertTrue(bool(np.isfinite(corners).all()))

    def test_deterministic_same_seed(self):
        a_img, a_c = make_synthetic_gate(seed=3)
        b_img, b_c = make_synthetic_gate(seed=3)
        self.assertTrue(torch.equal(a_img, b_img))
        np.testing.assert_array_equal(a_c, b_c)

    def test_corners_ordered_in_frame(self):
        # Corners are TL/TR/BR/BL in spec.K pixel coordinates (640x360).
        _, corners = make_synthetic_gate(seed=0)
        u = corners[:, 0]
        v = corners[:, 1]
        self.assertGreater(u.min(), 0)
        self.assertLess(u.max(), 640)
        self.assertGreater(v.min(), 0)
        self.assertLess(v.max(), 360)


class UntrainedUNetForwardTests(unittest.TestCase):
    def test_forward_output_shape_untrained(self):
        img, _ = make_synthetic_gate(seed=0)
        with torch.no_grad():
            out = UNet()(img)
        self.assertEqual(tuple(out.shape), (1, 1, TRAIN_H, TRAIN_W))
        self.assertTrue(bool(torch.isfinite(out).all()))


class PnPRecoveryTests(unittest.TestCase):
    def test_synthetic_corners_yield_valid_pose(self):
        expected_ranges = {
            0: 8.784991416560368,
            1: 8.540660644840012,
            2: 8.052055070103526,
            5: 9.134938152331644,
        }
        for seed, expected_range in expected_ranges.items():
            _, corners = make_synthetic_gate(seed=seed)
            estimate = estimate_pose(corners)
            self.assertIsNotNone(
                estimate, "PnP should recover a pose from synthetic corners"
            )
            self.assertAlmostEqual(
                estimate["range_m"], expected_range, delta=RANGE_TOL_M
            )
            self.assertLess(estimate["reproj_err_px"], 1.0)


class MissingWeightsTests(unittest.TestCase):
    def test_gate_pose_from_image_warns_and_returns_none_when_weights_absent(self):
        if os.path.isfile(WEIGHTS_PATH):
            self.skipTest("gatenet.pt present; missing-weight path not exercisable")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = gate_pose_from_image(np.zeros((360, 640, 3), np.uint8))
        self.assertIsNone(result)
        self.assertTrue(
            any(issubclass(w.category, RuntimeWarning) for w in caught),
            "expected a RuntimeWarning about missing GateNet weights",
        )


if __name__ == "__main__":
    unittest.main()
