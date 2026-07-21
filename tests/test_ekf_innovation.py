"""Offline units for ESKF Mahalanobis NIS innovation gate (no sim).

accept iff (z - h(xhat))^T S^{-1} (z - h(xhat)) < CHI2_3
with S = H P H^T + R. Rejected measurements must not change p or P.
"""

from __future__ import annotations

import unittest

import numpy as np

from rl.ekf import CHI2_3, ESKF


class InnovationGateTests(unittest.TestCase):
    def test_consistent_meas_accepted(self):
        ekf = ESKF(p0=np.zeros(3), p0_std=1.0, sigma_pos=0.5)
        p_before = ekf.p.copy()
        z = np.array([0.2, -0.1, 0.05])  # near prior
        ok = ekf.update_position(z, sigma=0.5, gate=True)
        self.assertTrue(ok)
        self.assertEqual(ekf.n_rejected, 0)
        # Filter should pull toward the measurement.
        self.assertLess(np.linalg.norm(ekf.p - z), np.linalg.norm(p_before - z))

    def test_wild_outlier_rejected_state_unchanged(self):
        ekf = ESKF(p0=np.zeros(3), p0_std=0.2, sigma_pos=0.3)
        p0 = ekf.p.copy()
        P0 = ekf.P.copy()
        z = np.array([50.0, -40.0, 30.0])  # huge vs P,R
        ok = ekf.update_position(z, sigma=0.3, gate=True)
        self.assertFalse(ok)
        self.assertEqual(ekf.n_rejected, 1)
        np.testing.assert_allclose(ekf.p, p0)
        np.testing.assert_allclose(ekf.P, P0)

    def test_gate_off_accepts_wild_meas(self):
        # Ungated path (e.g. callers that opt out) still applies the update.
        ekf = ESKF(p0=np.zeros(3), p0_std=0.2, sigma_pos=0.3)
        z = np.array([50.0, -40.0, 30.0])
        ok = ekf.update_position(z, sigma=0.3, gate=False)
        self.assertTrue(ok)
        self.assertEqual(ekf.n_rejected, 0)
        self.assertGreater(np.linalg.norm(ekf.p), 1.0)

    def test_nis_formula_boundary(self):
        # Hand-check: isotropic P_pos = s_p^2 I, R = s_r^2 I =>
        # S = (s_p^2 + s_r^2) I, nu = ||r||^2 / (s_p^2 + s_r^2).
        sp, sr = 1.0, 1.0
        ekf = ESKF(p0=np.zeros(3), p0_std=sp, sigma_pos=sr)
        # Set position cov exactly; zero cross terms already.
        ekf.P[:, :] = 0.0
        ekf.P[0:3, 0:3] = (sp**2) * np.eye(3)
        # nu just under CHI2_3 => accept
        # ||r||^2 / 2 < 7.815 => ||r|| < sqrt(15.63) ≈ 3.95
        r_ok = np.array([3.0, 0.0, 0.0])
        nu_ok = (r_ok @ r_ok) / (sp**2 + sr**2)
        self.assertLess(nu_ok, CHI2_3)
        self.assertTrue(ekf.update_position(r_ok, sigma=sr, gate=True))

        ekf2 = ESKF(p0=np.zeros(3), p0_std=sp, sigma_pos=sr)
        ekf2.P[:, :] = 0.0
        ekf2.P[0:3, 0:3] = (sp**2) * np.eye(3)
        r_bad = np.array([5.0, 0.0, 0.0])  # nu = 12.5 > 7.815
        nu_bad = (r_bad @ r_bad) / (sp**2 + sr**2)
        self.assertGreater(nu_bad, CHI2_3)
        self.assertFalse(ekf2.update_position(r_bad, sigma=sr, gate=True))


if __name__ == "__main__":
    unittest.main()
