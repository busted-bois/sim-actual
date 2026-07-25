"""Offline units for ESKF covariance time update: Pk^- = F Pk-1 F^T + Q."""

from __future__ import annotations

import unittest

import numpy as np

from rl.ekf import ESKF as RlESKF
from rl.ekf import skew as rl_skew
from rl.spec import quat_to_R
from simulator.state_estimator import ESKF as SimESKF
from simulator.state_estimator import skew as sim_skew


def _expected_P(P0, q, accel, gyro, dt, sa, sg, skew_fn):
    """Rebuild F,Q the same way ESKF.predict does (R from pre-predict q)."""
    accel = np.asarray(accel, float)
    gyro = np.asarray(gyro, float)
    R = quat_to_R(q)
    A = np.zeros((9, 9))
    A[0:3, 3:6] = np.eye(3)
    A[3:6, 6:9] = -R @ skew_fn(accel)
    A[6:9, 6:9] = -skew_fn(gyro)
    F = np.eye(9) + A * dt
    Q = np.zeros((9, 9))
    Q[3:6, 3:6] = (sa**2) * dt * dt * np.eye(3)
    Q[6:9, 6:9] = (sg**2) * dt * dt * np.eye(3)
    return F @ P0 @ F.T + Q, F, Q


def _sym(P):
    return 0.5 * (P + P.T)


class CovariancePredictTests(unittest.TestCase):
    def _algebra_lock(self, ekf_cls, skew_fn):
        sa, sg, dt = 0.3, 0.02, 0.01
        ekf = ekf_cls(
            p0=np.zeros(3),
            v0=np.zeros(3),
            p0_std=0.5,
            sigma_accel=sa,
            sigma_gyro=sg,
        )
        # Distinct prior cov so F P F^T is nontrivial.
        rng = np.random.default_rng(0)
        P0 = rng.normal(size=(9, 9))
        P0 = P0 @ P0.T + 0.1 * np.eye(9)
        ekf.P = P0.copy()
        q0 = ekf.q.copy()
        accel = np.array([0.1, -0.2, -9.81])
        gyro = np.array([0.01, -0.02, 0.005])
        expected, _, _ = _expected_P(P0, q0, accel, gyro, dt, sa, sg, skew_fn)
        ekf.predict(accel, gyro, dt)
        np.testing.assert_allclose(ekf.P, _sym(expected), rtol=1e-12, atol=1e-12)

    def test_rl_algebra_lock_FPF_plus_Q(self):
        self._algebra_lock(RlESKF, rl_skew)

    def test_sim_algebra_lock_FPF_plus_Q(self):
        self._algebra_lock(SimESKF, sim_skew)

    def test_trace_grows_without_measurements(self):
        ekf = RlESKF(p0_std=0.1, sigma_accel=0.5, sigma_gyro=0.05)
        tr0 = float(np.trace(ekf.P))
        accel = np.array([0.0, 0.0, -9.81])
        gyro = np.zeros(3)
        for _ in range(50):
            ekf.predict(accel, gyro, 0.02)
        self.assertGreater(float(np.trace(ekf.P)), tr0)

    def test_sim_trace_grows_without_measurements(self):
        ekf = SimESKF(p0_std=0.1, sigma_accel=0.5, sigma_gyro=0.05)
        tr0 = float(np.trace(ekf.P))
        accel = np.array([0.0, 0.0, -9.81])
        gyro = np.zeros(3)
        for _ in range(50):
            ekf.predict(accel, gyro, 0.02)
        self.assertGreater(float(np.trace(ekf.P)), tr0)


if __name__ == "__main__":
    unittest.main()
