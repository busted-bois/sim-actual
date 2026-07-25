"""Offline units for GPEstimation ESKF covariance predict + landmark update."""

from __future__ import annotations

import math
import unittest

import numpy as np

from simulator.gp_estimation import (
    C_THRUST,
    GPEstimation,
    SIGMA_ACCEL,
    SIGMA_GYRO,
    THRUST_POWERED,
)
from simulator.state_estimator import quat_from_rpy


class GPEstimationESKFTests(unittest.TestCase):
    def test_predict_grows_covariance_trace(self):
        est = GPEstimation({})
        est.set_thrust(0.27)
        tr0 = float(np.trace(est.ekf.P))
        imu = {
            "time_us": 0,
            "gx": 0.0,
            "gy": 0.0,
            "gz": 0.0,
            "ax": 0.0,
            "ay": 0.0,
            "az": -9.81,
        }
        est._process_imu(imu)  # seeds last ts
        for i in range(1, 61):
            imu["time_us"] = i * 10_000  # 100 Hz
            est._process_imu(imu)
        self.assertGreater(float(np.trace(est.ekf.P)), tr0)

    def test_unpowered_holds_velocity(self):
        est = GPEstimation({})
        est.set_thrust(0.0)
        est.ekf.v[:] = np.array([3.0, 0.0, 0.0])
        imu = {
            "time_us": 0,
            "gx": 0.0,
            "gy": 0.0,
            "gz": 0.0,
            "ax": 0.0,
            "ay": 0.0,
            "az": -9.81,
        }
        est._process_imu(imu)
        imu["time_us"] = 10_000
        est._process_imu(imu)
        np.testing.assert_allclose(est.ekf.v, 0.0, atol=1e-9)

    def test_landmark_pulls_position(self):
        est = GPEstimation({})
        est.ekf.p[:] = np.array([1.0, 0.0, 0.0])
        est.ekf.P = np.eye(9) * 4.0  # uncertain → trusts measurement
        ok = est.update_landmark(np.array([0.0, 0.0, 0.0]), range_m=2.0)
        self.assertTrue(ok)
        self.assertLess(float(np.linalg.norm(est.ekf.p)), 0.8)
        self.assertEqual(est.n_landmarks, 1)

    def test_zero_velocity_keeps_position(self):
        est = GPEstimation({})
        est.ekf.p[:] = np.array([5.0, 1.0, -0.5])
        est.ekf.v[:] = np.array([2.0, 0.0, 0.0])
        est.pos_ned[:] = est.ekf.p
        est.zero_velocity()
        np.testing.assert_allclose(est.ekf.p, [5.0, 1.0, -0.5])
        np.testing.assert_allclose(est.ekf.v, 0.0)

    def test_sigma_defaults_match_module(self):
        est = GPEstimation({})
        self.assertEqual(est.ekf.sa, SIGMA_ACCEL)
        self.assertEqual(est.ekf.sg, SIGMA_GYRO)
        self.assertGreater(C_THRUST, THRUST_POWERED)
        q0 = quat_from_rpy(0.0, math.radians(-17.8), 0.0)
        np.testing.assert_allclose(est.ekf.q, q0, atol=1e-9)


if __name__ == "__main__":
    unittest.main()
