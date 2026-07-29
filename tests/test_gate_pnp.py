"""gate_pnp: 4-corner CV path + reprojection rejection."""

import unittest

import cv2
import numpy as np

from simulator.gate_corners_cv import find_gate_inner_corners, order_corners
from simulator.gate_pnp import (
    _DIST,
    _GATE_PTS_3D,
    _INNER_CORNERS_3D,
    _K,
    MAX_REPROJ_PX,
    estimate_gate_pose,
    estimate_gate_pose_from_corners,
    estimate_pose,
)


def _project(pts3d, rvec, tvec):
    proj, _ = cv2.projectPoints(
        np.asarray(pts3d, np.float64),
        np.asarray(rvec, float),
        np.asarray(tvec, float),
        _K,
        _DIST,
    )
    return proj.reshape(-1, 2)


class CornerPnPTests(unittest.TestCase):
    def test_recovers_known_pose(self):
        rng = np.random.default_rng(0)
        for _ in range(50):
            t_true = np.array(
                [rng.uniform(-1, 1), rng.uniform(-0.5, 0.5), rng.uniform(3, 9)]
            )
            r_true = np.array([rng.uniform(-0.2, 0.2), rng.uniform(-0.2, 0.2), 0.0])
            corners = _project(_INNER_CORNERS_3D, r_true, t_true)
            est = estimate_gate_pose_from_corners(corners)
            self.assertIsNotNone(est)
            self.assertEqual(est["method"], "ippe-cv4")
            self.assertLess(np.linalg.norm(est["gate_pos_cam"] - t_true), 0.1)
            self.assertLess(est["reproj_px"], 1.0)

    def test_end_to_end_synthetic_render(self):
        # Render the gate ring at a known pose, extract CV corners, solve PnP,
        # compare against the ground-truth camera-frame position.
        t_true = np.array([0.3, -0.2, 6.0])
        r_true = np.zeros(3)
        # _GATE_PTS_3D outer slots are in YOLO order, not polygon order --
        # sort into a drawable quad before rasterizing.
        outer_px = order_corners(_project(_GATE_PTS_3D[4:], r_true, t_true))
        inner_px = _project(_INNER_CORNERS_3D, r_true, t_true)
        img = np.full((360, 640, 3), 30, np.uint8)
        cv2.fillPoly(img, [outer_px.astype(np.int32)], (15, 57, 243))
        cv2.fillPoly(img, [inner_px.astype(np.int32)], (30, 30, 30))
        corners = find_gate_inner_corners(img)
        self.assertIsNotNone(corners)
        est = estimate_gate_pose_from_corners(corners)
        self.assertIsNotNone(est)
        self.assertLess(np.linalg.norm(est["gate_pos_cam"] - t_true), 0.3)

    def test_behind_camera_rejected(self):
        corners = _project(_INNER_CORNERS_3D, np.zeros(3), np.array([0, 0, 5.0]))
        # A mirrored quad (TL<->TR, BL<->BR) is not a rigid view of the gate.
        mirrored = corners[[1, 0, 3, 2]]
        self.assertIsNone(estimate_gate_pose_from_corners(mirrored))

    def test_nonfinite_corners_rejected(self):
        corners = _project(_INNER_CORNERS_3D, np.zeros(3), np.array([0, 0, 5.0]))
        corners[0, 0] = np.nan
        self.assertIsNone(estimate_gate_pose_from_corners(corners))


class ReprojRejectTests(unittest.TestCase):
    def test_scrambled_keypoints_rejected(self):
        # Wrong correspondence on the over-determined 8-point solve must blow
        # past MAX_REPROJ_PX and be rejected instead of yielding a bogus pose.
        t_true = np.array([0.2, 0.1, 5.0])
        kp = _project(_GATE_PTS_3D, np.zeros(3), t_true)
        confs = np.ones(len(kp))
        self.assertIsNotNone(estimate_pose(kp, confs))
        scrambled = kp.copy()
        scrambled[[0, 3]] = scrambled[[3, 0]]  # swap TL-inner / BR-inner
        self.assertIsNone(estimate_pose(scrambled, confs))
        self.assertIsNone(estimate_gate_pose(scrambled, confs))

    def test_threshold_value(self):
        self.assertEqual(MAX_REPROJ_PX, 10.0)


if __name__ == "__main__":
    unittest.main()
