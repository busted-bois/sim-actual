import math
import unittest

import numpy as np

from rl.vision_fusion import fuse_gate_bearing_yaw, fuse_gate_target_position
from rl.ekf import ESKF
from simulator.transforms import quat_to_yaw
from simulator.vq2_pose import (
    VQ2PoseEstimator,
    gate_z_ned,
    spawn_heading_ned,
    spawn_position_ned,
)


class VQ2PoseTests(unittest.TestCase):
    def test_spawn_before_gate_zero(self):
        gate_map = [{"pos": [-20.0, 0.0, 0.0]}]
        p = spawn_position_ned(gate_map)
        self.assertGreater(p[0], -20.0)

    def test_spawn_faces_gate_zero(self):
        """Spawn heading must point toward gate 0, not the hardcoded identity."""
        gate_map = [{"pos": [-20.0, 0.0, 0.0]}]
        yaw0 = spawn_heading_ned(gate_map)
        self.assertAlmostEqual(abs(yaw0), math.pi, delta=1e-6)

    def test_gate_z_ned_flip(self):
        pos = [0.0, 0.0, 5.0]
        self.assertEqual(gate_z_ned(pos, flipz=False), 5.0)
        self.assertEqual(gate_z_ned(pos, flipz=True), -5.0)

    def test_spawn_position_uses_flipz(self):
        # Climb-course map (z up-positive): gate 0 near ground (z~=0).
        gate_map = [{"pos": [-20.0, 0.0, -0.03]}, {"pos": [-40.0, 0.0, 5.0]}]
        p_flat = spawn_position_ned(gate_map, flipz=False)
        p_climb = spawn_position_ned(gate_map, flipz=True)
        self.assertNotAlmostEqual(p_flat[2], p_climb[2])
        # flipz=True normalizes to NED (down-positive); spawn should be
        # "above" gate 0, i.e. more negative than gate 0's normalized z.
        self.assertLess(p_climb[2], gate_z_ned(gate_map[0]["pos"], True))

    def test_reset_initializes_yaw_toward_gate_zero(self):
        est = VQ2PoseEstimator()
        gate_map = [{"pos": [-20.0, 0.0, 0.0]}]
        est.reset(gate_map)
        yaw0 = quat_to_yaw(*est.ekf.q)
        self.assertAlmostEqual(abs(yaw0), math.pi, delta=1e-6)

    def test_imu_predict_updates_state(self):
        est = VQ2PoseEstimator()
        est.reset([{"pos": [0.0, 0.0, -5.0]}])
        data = {
            "imu": {
                "time_us": 0,
                "ax": 0.0,
                "ay": 0.0,
                "az": 0.0,
                "gx": 0.0,
                "gy": 0.0,
                "gz": 0.0,
                "pressure_alt": 100.0,
            },
            "active_gate_index": 0,
        }
        odo0 = est.tick(data, [{"pos": [10.0, 0.0, -5.0]}])
        self.assertIsNotNone(odo0)
        data["imu"]["time_us"] = 10000
        data["imu"]["ax"] = 1.0
        odo1 = est.tick(data, [{"pos": [10.0, 0.0, -5.0]}])
        self.assertIsNotNone(odo1)
        self.assertNotEqual(odo0["vx"], odo1["vx"])

    def test_vision_fusion_moves_ekf_position(self):
        ekf = ESKF(p0=np.zeros(3), v0=np.zeros(3))
        gate_target = {
            "bearing_rad": 0.0,
            "elevation_rad": 0.0,
            "range_m": 10.0,
            "confidence": 0.6,
        }
        gate_world = np.array([10.0, 0.0, -5.0])
        q = np.array([1.0, 0.0, 0.0, 0.0])
        self.assertTrue(
            fuse_gate_target_position(ekf, gate_target, gate_world, q)
        )
        self.assertGreater(np.linalg.norm(ekf.p - np.zeros(3)), 0.5)

    def test_vision_fusion_uses_cam_to_body_not_body_to_cam(self):
        """A centered detection (bearing=elevation=0) is straight ahead in the
        body's forward axis (only tilted up by the camera's pitch), not off
        to the side. Regression test for the R_CAM_BODY/R_BODY_CAM mixup."""
        ekf = ESKF(p0=np.zeros(3), v0=np.zeros(3))
        gate_target = {
            "bearing_rad": 0.0,
            "elevation_rad": 0.0,
            "range_m": 10.0,
            "confidence": 1.0,
        }
        gate_world = np.array([10.0, 0.0, -3.0])
        q = np.array([1.0, 0.0, 0.0, 0.0])
        fuse_gate_target_position(ekf, gate_target, gate_world, q)
        # p_meas = gate_world - p_body; with q=identity, ekf.p should land
        # near y=0 (forward-only), not shifted sideways.
        self.assertAlmostEqual(ekf.p[1], 0.0, delta=1e-6)

    def test_fuse_gate_bearing_yaw_converges_toward_true_heading(self):
        """Repeated fusion (as happens every control tick) should pull yaw
        toward the true heading implied by gate geometry + camera bearing,
        even though the filter starts with a badly wrong (identity) yaw."""
        ekf = ESKF(p0=np.zeros(3), v0=np.zeros(3), q0=np.array([1.0, 0.0, 0.0, 0.0]))
        gate_target = {
            "detected": True,
            "range_m": 10.0,
            "confidence": 0.9,
            "bearing_rad": 0.0,
        }
        gate_world = np.array([-10.0, 0.0, 0.0])  # true yaw ~= pi, not 0
        for _ in range(50):
            ok = fuse_gate_bearing_yaw(ekf, gate_target, np.zeros(3), gate_world)
            self.assertTrue(ok)
        yaw = quat_to_yaw(*ekf.q)
        self.assertAlmostEqual(abs(yaw), math.pi, delta=0.3)

    def test_gate_target_dedup_skips_repeated_frame(self):
        """The same camera frame must not be fused as independent
        measurements on every 250Hz control tick."""
        est = VQ2PoseEstimator()
        gate_map = [{"pos": [10.0, 0.0, -5.0]}]
        est.reset(gate_map)
        gate_target = {
            "detected": True,
            "frame_id": 42,
            "bearing_rad": 0.0,
            "elevation_rad": 0.0,
            "range_m": 10.0,
            "confidence": 0.9,
        }
        data = {"gate_target": gate_target, "active_gate_index": 0}
        est.tick(data, gate_map)
        p_after_first = est.ekf.p.copy()
        est.tick(data, gate_map)
        np.testing.assert_allclose(est.ekf.p, p_after_first)

    def test_gate_target_new_frame_is_fused(self):
        est = VQ2PoseEstimator()
        gate_map = [{"pos": [10.0, 0.0, -5.0]}]
        est.reset(gate_map)
        base_target = {
            "detected": True,
            "bearing_rad": 0.0,
            "elevation_rad": 0.0,
            "range_m": 10.0,
            "confidence": 0.9,
        }
        data = {"gate_target": {**base_target, "frame_id": 1}, "active_gate_index": 0}
        est.tick(data, gate_map)
        p_after_first = est.ekf.p.copy()
        data["gate_target"] = {**base_target, "frame_id": 2}
        est.tick(data, gate_map)
        self.assertFalse(np.allclose(est.ekf.p, p_after_first))


if __name__ == "__main__":
    unittest.main()
