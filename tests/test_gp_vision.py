"""Tests for classical CV-hole vision fusion (Try 2)."""

from __future__ import annotations

import unittest

import numpy as np

from simulator.gp_vision import (
    GATE_INNER_W_M,
    GateEstimateSmoother,
    VisionVelocityTracker,
    best_pose_gate,
    cv_opening_pixels,
    gate_body_from_pinhole,
    vision_gate_estimate,
    yolo_opening_pixels,
)


def _yolo_gate(
    *,
    body=(8.0, -0.5, 0.2),
    conf=0.9,
    reproj=1.5,
    keypoints=None,
    keypoint_conf=None,
    box=None,
    cv_corners=None,
):
    g = {
        "conf": conf,
        "pose": {
            "gate_pos_body": np.array(body, dtype=np.float64),
            "normal_body": np.array([-1.0, 0.0, 0.0]),
            "reproj_px": reproj,
            "method": "ippe-yolo8",
        },
        "cv_corners": cv_corners,
    }
    if keypoints is not None:
        g["keypoints"] = keypoints
        g["keypoint_conf"] = keypoint_conf
    if box is not None:
        g["box"] = box
    return g


class CvOpeningPixelsTests(unittest.TestCase):
    def test_cv_corners_centroid_and_spread(self):
        corners = np.array(
            [[270.0, 140.0], [370.0, 140.0], [370.0, 220.0], [270.0, 220.0]]
        )
        u, v, s = cv_opening_pixels({"cv_corners": corners})
        self.assertAlmostEqual(u, 320.0, places=3)
        self.assertAlmostEqual(v, 180.0, places=3)
        self.assertAlmostEqual(s, 100.0, places=3)

    def test_yolo_opening_prefers_cv(self):
        corners = np.array(
            [[300.0, 160.0], [340.0, 160.0], [340.0, 200.0], [300.0, 200.0]]
        )
        g = _yolo_gate(
            box=[200.0, 100.0, 440.0, 260.0],
            cv_corners=corners,
            keypoints=np.zeros((8, 2)),
            keypoint_conf=np.ones(8),
        )
        u, v, s = yolo_opening_pixels(g)
        self.assertAlmostEqual(u, 320.0, places=3)
        self.assertAlmostEqual(v, 180.0, places=3)
        self.assertAlmostEqual(s, 40.0, places=3)


class CvHoleEstimateTests(unittest.TestCase):
    def test_cv_hole_wins_over_yolo_pnp_body(self):
        corners = np.array(
            [[270.0, 140.0], [370.0, 140.0], [370.0, 220.0], [270.0, 220.0]]
        )
        # Biased YOLO PnP body (would aim high / left); CV hole is centred.
        g = _yolo_gate(body=(5.0, 2.0, -1.5), cv_corners=corners)
        data = {
            "pose": {"gates": [g], "frame_id": 7},
            "anduril_gate": {
                "body_x_m": 9.0,
                "body_y_m": 1.0,
                "body_z_m": -0.5,
                "source": "anduril",
            },
        }
        est = vision_gate_estimate(data)
        self.assertIsNotNone(est)
        self.assertEqual(est["source"], "cv")
        self.assertTrue(est["opening_ok"])
        self.assertFalse(est["pnp_ok"])
        self.assertEqual(est["method"], "hsv-hole")
        self.assertAlmostEqual(est["u_px"], 320.0, places=3)
        self.assertAlmostEqual(est["v_px"], 180.0, places=3)
        self.assertAlmostEqual(est["inner_spread_px"], 100.0, places=3)
        # Body from inner 1.5 m pinhole, not YOLO (5,-1.5) or Anduril.
        expect = gate_body_from_pinhole(320.0, 180.0, 100.0, gate_w_m=GATE_INNER_W_M)
        self.assertIsNotNone(expect)
        self.assertAlmostEqual(est["body_x_m"], expect[0], places=4)
        self.assertAlmostEqual(est["body_y_m"], expect[1], places=4)
        self.assertAlmostEqual(est["body_z_m"], expect[2], places=4)
        self.assertNotAlmostEqual(est["body_z_m"], -1.5, places=1)

    def test_inner_pinhole_scale(self):
        body_inner = gate_body_from_pinhole(320.0, 180.0, 100.0, gate_w_m=1.5)
        body_outer = gate_body_from_pinhole(320.0, 180.0, 100.0, gate_w_m=2.7)
        self.assertIsNotNone(body_inner)
        self.assertIsNotNone(body_outer)
        self.assertLess(body_inner[0], body_outer[0])

    def test_yolo_fallback_without_cv(self):
        g = _yolo_gate(body=(8.0, -0.5, 0.2))
        est = vision_gate_estimate({"pose": {"gates": [g], "frame_id": 1}})
        self.assertEqual(est["source"], "yolo")
        self.assertTrue(est["opening_ok"])
        self.assertAlmostEqual(est["body_x_m"], 8.0, places=4)

    def test_anduril_not_opening_ok(self):
        data = {
            "anduril_gate": {
                "body_x_m": 4.0,
                "body_y_m": 0.0,
                "body_z_m": 0.0,
                "source": "anduril",
            }
        }
        est = vision_gate_estimate(data)
        self.assertEqual(est["source"], "anduril")
        self.assertFalse(est["opening_ok"])


class BestPoseGateTests(unittest.TestCase):
    def test_rejects_low_conf(self):
        g = _yolo_gate(conf=0.2)
        self.assertIsNone(best_pose_gate({"pose": {"gates": [g]}}))


class SmootherTests(unittest.TestCase):
    def test_cv_not_stuck_behind_yolo_sticky(self):
        sm = GateEstimateSmoother()
        yolo_g = _yolo_gate(body=(8.0, 0.0, 0.0))
        corners = np.array(
            [[270.0, 140.0], [370.0, 140.0], [370.0, 220.0], [270.0, 220.0]]
        )
        cv_g = _yolo_gate(body=(5.0, 0.0, -1.0), cv_corners=corners)
        sm.update({"pose": {"gates": [yolo_g], "frame_id": 1}})
        out = sm.update({"pose": {"gates": [cv_g], "frame_id": 2}})
        self.assertEqual(out["source"], "cv")
        self.assertTrue(out["opening_ok"])


class VisionVelocityTests(unittest.TestCase):
    def test_velocity_from_delta(self):
        tr = VisionVelocityTracker()
        tr.update(
            {"frame_id": 1, "body_x_m": 10.0, "body_y_m": 0.0, "body_z_m": 0.0},
            cam_hz=30.0,
        )
        vel = tr.update(
            {"frame_id": 2, "body_x_m": 9.0, "body_y_m": 0.0, "body_z_m": 0.0},
            cam_hz=30.0,
        )
        self.assertIsNotNone(vel)
        # Closing 1 m in 1/30 s → +30 m/s forward OF convention.
        self.assertAlmostEqual(vel["vx_body_mps"], 30.0, places=3)


if __name__ == "__main__":
    unittest.main()
