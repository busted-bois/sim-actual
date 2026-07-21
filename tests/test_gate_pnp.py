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


class OneSidedVisibilityTests(unittest.TestCase):
    def test_center_unbiased_when_bottom_corners_clipped(self):
        # The 20-deg-up camera pushes the BOTTOM corners out of frame inside
        # ~4.5 m. The aim point must stay the opening CENTRE (from the PnP
        # solve), not the centroid of the surviving top corners (+0.75 m up —
        # the drone flew to that and clipped the top bar).
        # Object +y projects DOWNWARD in the image, so the image-bottom
        # corners (the ones the tilt crops first) are slots {0,1} inner and
        # {4,5} outer.
        t_true = np.array([0.1, 0.85, 3.0])
        kp = _project(_GATE_PTS_3D, np.zeros(3), t_true)
        confs = np.ones(8)
        confs[[0, 1, 4, 5]] = 0.0  # image-bottom inner + outer "out of frame"
        est = estimate_gate_pose(kp, confs)
        self.assertIsNotNone(est)
        np.testing.assert_allclose(est["gate_pos_cam"], t_true, atol=0.05)

    def test_center_unbiased_when_side_corners_clipped(self):
        # Same defense laterally (gate half out of frame while banking):
        # drop one x-column of corners (slots {1,3} inner, {5,7} outer).
        t_true = np.array([-0.9, 0.1, 3.0])
        kp = _project(_GATE_PTS_3D, np.zeros(3), t_true)
        confs = np.ones(8)
        confs[[1, 3, 5, 7]] = 0.0
        est = estimate_gate_pose(kp, confs)
        self.assertIsNotNone(est)
        np.testing.assert_allclose(est["gate_pos_cam"], t_true, atol=0.05)

    def test_balanced_corners_keep_centroid_path(self):
        # Full visibility must keep the original centroid behavior exactly.
        t_true = np.array([0.3, -0.2, 6.0])
        kp = _project(_GATE_PTS_3D, np.zeros(3), t_true)
        est = estimate_gate_pose(kp, np.ones(8))
        self.assertIsNotNone(est)
        np.testing.assert_allclose(est["gate_pos_cam"], t_true, atol=0.05)

    def test_diagonal_pair_is_balanced(self):
        t_true = np.array([0.0, 0.2, 4.0])
        kp = _project(_GATE_PTS_3D, np.zeros(3), t_true)
        confs = np.ones(8)
        confs[[1, 2]] = 0.0  # keep TL+BR inner: diagonal pair, still centred
        est = estimate_gate_pose(kp, confs)
        self.assertIsNotNone(est)
        np.testing.assert_allclose(est["gate_pos_cam"], t_true, atol=0.05)


class EdgePairFallbackTests(unittest.TestCase):
    # Inside ~4.5 m the crop leaves <4 confident corners and full PnP is
    # impossible. The pose must survive off a single visible inner edge (the
    # bar under the gate's "AI GP" banner) or the pilot loses the CURRENT
    # gate at 3-5 m out and retargets a far one -- log-verified as the
    # dominant gate-2+ frame clip.

    def test_pose_survives_two_corner_crop_with_box(self):
        t_true = np.array([0.2, 0.6, 2.5])
        kp = _project(_GATE_PTS_3D, np.zeros(3), t_true)
        confs = np.zeros(8)
        confs[[2, 3]] = 1.0  # one inner edge left in frame
        box = np.array(
            [kp[:, 0].min(), kp[:, 1].min(), kp[:, 0].max(), kp[:, 1].max()]
        )
        est = estimate_gate_pose(kp, confs, box=box)
        self.assertIsNotNone(est)
        self.assertEqual(est["method"], "edge-pair")
        np.testing.assert_allclose(est["gate_pos_cam"], t_true, atol=0.15)

    def test_pose_survives_three_corner_crop_via_outer_corner(self):
        # No box: the same-edge OUTER corner pins the inward direction.
        t_true = np.array([-0.1, 0.7, 2.2])
        kp = _project(_GATE_PTS_3D, np.zeros(3), t_true)
        confs = np.zeros(8)
        confs[[2, 3, 6]] = 1.0
        est = estimate_gate_pose(kp, confs)
        self.assertIsNotNone(est)
        self.assertEqual(est["method"], "edge-pair")
        np.testing.assert_allclose(est["gate_pos_cam"], t_true, atol=0.15)

    def test_single_corner_still_rejected(self):
        kp = _project(_GATE_PTS_3D, np.zeros(3), np.array([0.0, 0.0, 3.0]))
        confs = np.zeros(8)
        confs[2] = 1.0
        self.assertIsNone(estimate_gate_pose(kp, confs))

    def test_bar_hugging_box_cannot_flip_inward_sign(self):
        # One horizontal inner edge visible with the YOLO box hugging the bar:
        # the box centre sits ~0.2*edge off the edge on the WRONG side (toward
        # the outer ring) — trusting it aimed the drone 1.45 m below a gate at
        # 3 m. Any other visible corner is a raw-pixel cue that must win.
        t_true = np.array([0.1, 0.7, 3.0])
        kp = _project(_GATE_PTS_3D, np.zeros(3), t_true)
        confs = np.zeros(8)
        confs[[2, 3, 1]] = 1.0  # one inner edge + one opposite inner corner
        edge_v = kp[2, 1]
        outer_v = kp[6, 1]  # same-corner outer: the bar side
        x0, x1 = kp[[2, 3], 0].min(), kp[[2, 3], 0].max()
        bar_box = np.array(
            [x0, min(edge_v, outer_v), x1, max(edge_v, outer_v)]
        )
        est = estimate_gate_pose(kp, confs, box=bar_box)
        self.assertIsNotNone(est)
        self.assertEqual(est["method"], "edge-pair")
        np.testing.assert_allclose(est["gate_pos_cam"], t_true, atol=0.2)


class DiagonalCentreTests(unittest.TestCase):
    def test_oblique_view_centre_unbiased(self):
        # Perspective pulls the 4-corner centroid toward the nearer edge
        # (6-10% of the offset, always damping convergence toward centre);
        # the diagonal intersection is projective-exact.
        t_true = np.array([0.1, -0.8, 3.0])
        r_true = np.array([0.45, 0.0, 0.0])
        kp = _project(_GATE_PTS_3D, r_true, t_true)
        est = estimate_gate_pose(kp, np.ones(8))
        self.assertIsNotNone(est)
        np.testing.assert_allclose(est["gate_pos_cam"], t_true, atol=0.05)


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
