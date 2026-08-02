"""Deterministic tests for ESKF closed-loop: thrust-model vs raw-accel, coast,
reset, collision, adaptive gate, sign variants, dropout recovery, fusion."""

from __future__ import annotations

import math
import unittest

import numpy as np
from numpy.testing import assert_allclose

from rl.core.spec import quat_to_R
from rl.estimation.ekf import (
    C_THRUST,
    DRAG_KD,
    ESKF,
    GRAVITY,
    GYRO_SPIKE,
)
from rl.perception.vision_fusion import (
    fuse_pnp_gate,
    step_pnp_fusion,
)


def _hover_thrust() -> float:
    return GRAVITY / C_THRUST


def _level_q() -> np.ndarray:
    return np.array([1.0, 0.0, 0.0, 0.0])


def _good_det(gate_body, conf=0.9, range_m=5.0, reproj=1.0):
    return {
        "conf": conf,
        "pose": {
            "gate_pos_body": list(gate_body),
            "range_m": range_m,
            "reproj_px": reproj,
        },
    }


def _gate_world_at(x, y, z):
    return np.array([x, y, z])


class _Base(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(42)
        self.dt = 1 / 100.0
        self.q0 = _level_q()
        self.R0 = quat_to_R(self.q0)

    def _garbage_accel(self) -> np.ndarray:
        return self.rng.uniform(-400, 400, 3)

    def _clean_gyro(self) -> np.ndarray:
        return self.rng.normal(0, 0.002, 3)


class TestCommandedPredictBeatsRawAccel(_Base):
    def test_final_window_error_under_05m(self):
        dt, T = self.dt, 20.0
        n = int(T / dt)
        v0 = np.array([2.0, 1.0, -0.5])
        kd = DRAG_KD
        hover = _hover_thrust()

        def p_true(t):
            return v0 / kd * (1.0 - math.exp(-kd * t))

        ekf_raw = ESKF(sigma_accel=0.3)
        ekf_cmd = ESKF(sigma_accel=2.0)
        errs_raw, errs_cmd = [], []

        for k in range(n):
            t = k * dt
            garbage = self._garbage_accel()
            gyro = self._clean_gyro()
            ekf_raw.predict(garbage, gyro, dt)
            ekf_cmd.predict_commanded(hover, gyro, dt)
            pt = p_true(t)
            errs_raw.append(np.linalg.norm(ekf_raw.p - pt))
            errs_cmd.append(np.linalg.norm(ekf_cmd.p - pt))
            if t > T / 2 and k % 20 == 0:
                ekf_cmd.update_position(pt + self.rng.normal(0, 0.15, 3))

        window = 150
        final_cmd = float(np.mean(errs_cmd[-window:]))
        final_raw = float(np.mean(errs_raw[-window:]))
        assert final_cmd < 0.5, f"cmd pos err {final_cmd:.2f} >= 0.5m"
        assert final_cmd < final_raw, f"cmd {final_cmd:.2f} not < raw {final_raw:.2f}"


class TestCoast(_Base):
    def test_state_unchanged_trace_grows(self):
        ekf = ESKF(p0=np.array([1, 2, 3]), v0=np.array([0.1, 0, -0.1]), q0=self.q0)
        p0, v0, q0 = ekf.p.copy(), ekf.v.copy(), ekf.q.copy()
        tr0 = ekf.state()["P_trace"]
        ekf.coast(10, self.dt)
        assert_allclose(ekf.p, p0, atol=1e-15)
        assert_allclose(ekf.v, v0, atol=1e-15)
        assert_allclose(ekf.q, q0, atol=1e-15)
        self.assertGreater(ekf.state()["P_trace"], tr0)

    def test_bad_dt_is_noop(self):
        ekf = ESKF()
        tr0 = ekf.state()["P_trace"]
        ekf.coast(5, 0.0)
        ekf.coast(5, -1.0)
        ekf.coast(5, 1.0)
        ekf.coast(0, self.dt)
        self.assertAlmostEqual(ekf.state()["P_trace"], tr0)

    def test_nonfinite_covariance_does_not_emit_runtime_warning(self):
        ekf = ESKF()
        ekf.P[0, 0] = float("inf")
        with np.errstate(all="raise"):
            ekf.coast(1, self.dt)
        self.assertEqual(ekf.coast_count, 1)
        self.assertFalse(ekf.healthy())

    def test_coast_count_increments(self):
        ekf = ESKF(q0=self.q0)
        self.assertEqual(ekf.coast_count, 0)
        ekf.coast(7, self.dt)
        self.assertEqual(ekf.coast_count, 7)

    def test_coast_count_cleared_by_update(self):
        ekf = ESKF(q0=self.q0)
        ekf.coast(5, self.dt)
        self.assertEqual(ekf.coast_count, 5)
        ekf.update_position(np.zeros(3), sigma=0.5)
        self.assertEqual(ekf.coast_count, 0)


class TestRecoveryCount(_Base):
    def test_reset_increments(self):
        ekf = ESKF(q0=self.q0)
        self.assertEqual(ekf.recovery_count, 0)
        ekf.reset()
        self.assertEqual(ekf.recovery_count, 1)
        ekf.reset()
        self.assertEqual(ekf.recovery_count, 2)

    def test_reset_on_detection_increments(self):
        ekf = ESKF(q0=self.q0, p0=np.array([5, 5, 5]))
        self.assertEqual(ekf.recovery_count, 0)
        ekf.reset_on_detection(drone_pos=np.zeros(3))
        self.assertEqual(ekf.recovery_count, 1)
        self.assertEqual(ekf.coast_count, 0)


class TestReset(_Base):
    def test_strict_reset_zeros(self):
        ekf = ESKF(p0=np.array([99, 99, 99]), v0=np.array([10, 0, 0]))
        ekf.reset()
        assert_allclose(ekf.p, 0)
        assert_allclose(ekf.v, 0)
        assert_allclose(ekf.q, [1, 0, 0, 0])

    def test_reset_honors_v0(self):
        ekf = ESKF()
        ekf.reset(v0=np.array([1, 2, 3]))
        assert_allclose(ekf.v, [1, 2, 3])

    def test_reset_validates_v0(self):
        ekf = ESKF()
        with self.assertRaises(ValueError):
            ekf.reset(v0="not_an_array")

    def test_reset_validates_p0(self):
        ekf = ESKF()
        with self.assertRaises(ValueError):
            ekf.reset(p0=np.array([float("nan"), 0, 0]))

    def test_reset_validates_p0_std(self):
        ekf = ESKF(p0=np.array([1, 2, 3]), v0=np.array([4, 5, 6]))
        p_before, v_before, q_before, P_before = (
            ekf.p.copy(),
            ekf.v.copy(),
            ekf.q.copy(),
            ekf.P.copy(),
        )
        with self.assertRaises(ValueError):
            ekf.reset(p0=np.zeros(3), p0_std=-1.0)
        with self.assertRaises(ValueError):
            ekf.reset(p0_std=float("nan"))
        assert_allclose(ekf.p, p_before)
        assert_allclose(ekf.v, v_before)
        assert_allclose(ekf.q, q_before)
        assert_allclose(ekf.P, P_before)


class TestResetOnDetection(_Base):
    def test_drone_pos_path(self):
        ekf = ESKF(p0=np.array([5, 5, 5]))
        ekf.reset_on_detection(drone_pos=np.array([1, 2, 3]))
        assert_allclose(ekf.p, [1, 2, 3])
        assert_allclose(ekf.v, 0)

    def test_pnp_path(self):
        ekf = ESKF(p0=np.array([5, 5, 5]), q0=self.q0)
        gate_world = _gate_world_at(10, 0, -3)
        gate_body = np.array([0, 0, 5])
        ekf.reset_on_detection(pnp_gate_body=gate_body, gate_world_pos=gate_world)
        expected = gate_world - quat_to_R(self.q0) @ gate_body
        assert_allclose(ekf.p, expected)
        assert_allclose(ekf.v, 0)

    def test_preserves_q_when_omitted(self):
        yaw_q = _quat_from_rpy_yaw(math.radians(45))
        ekf = ESKF(q0=yaw_q)
        ekf.reset_on_detection(drone_pos=np.zeros(3))
        assert_allclose(ekf.q, yaw_q, atol=1e-15)

    def test_accepts_explicit_q(self):
        ekf = ESKF(q0=self.q0)
        new_q = _quat_from_rpy_yaw(math.radians(-90))
        ekf.reset_on_detection(drone_pos=np.zeros(3), q=new_q)
        assert_allclose(ekf.q, new_q, atol=1e-15)

    def test_rejects_nonfinite_q(self):
        ekf = ESKF()
        with self.assertRaises(ValueError):
            ekf.reset_on_detection(
                drone_pos=np.zeros(3), q=np.array([1, 0, 0, float("nan")])
            )

    def test_ambiguous_rejects(self):
        ekf = ESKF()
        with self.assertRaises(ValueError):
            ekf.reset_on_detection(
                drone_pos=np.zeros(3),
                pnp_gate_body=np.zeros(3),
                gate_world_pos=np.zeros(3),
            )

    def test_missing_rejects(self):
        ekf = ESKF()
        with self.assertRaises(ValueError):
            ekf.reset_on_detection()

    def test_nonfinite_rejects(self):
        ekf = ESKF()
        with self.assertRaises(ValueError):
            ekf.reset_on_detection(drone_pos=np.array([float("nan"), 0, 0]))

    def test_wrong_shape_rejects(self):
        ekf = ESKF()
        with self.assertRaises(ValueError):
            ekf.reset_on_detection(drone_pos=np.array([1, 2]))


def _quat_from_rpy_yaw(yaw):
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
    return np.array([cy, 0, 0, sy])


class TestCollision(_Base):
    def test_velocity_zeroed_cov_opened(self):
        ekf = ESKF(p0=np.zeros(3), v0=np.array([5, 0, 0]), q0=self.q0)
        tr0 = ekf.state()["P_trace"]
        ekf.notify_collision()
        assert_allclose(ekf.v, 0)
        self.assertGreater(ekf.state()["P_trace"], tr0)

    def test_position_cov_increased(self):
        ekf = ESKF(q0=self.q0, p0_std=0.1)
        pp_before = float(np.trace(ekf.P[0:3, 0:3]))
        ekf.notify_collision()
        pp_after = float(np.trace(ekf.P[0:3, 0:3]))
        self.assertGreater(pp_after - pp_before, 24.0)


class TestUpdateShrinksTrace(_Base):
    def test_position_update_reduces_trace(self):
        ekf = ESKF(p0=np.zeros(3), q0=self.q0, p0_std=2.0)
        ekf.coast(50, self.dt)
        tr_before = ekf.state()["P_trace"]
        ekf.update_position(np.zeros(3), sigma=0.3)
        self.assertLess(ekf.state()["P_trace"], tr_before)


class TestAdaptiveGate(_Base):
    def test_gate_grows_with_uncertainty(self):
        ekf = ESKF(p0_std=1.0)
        g1 = ekf.innovation_gate(floor=1.0)
        ekf.coast(200, self.dt)
        g2 = ekf.innovation_gate(floor=1.0)
        self.assertGreater(g2, g1)
        self.assertGreaterEqual(g1, 1.0)

    def test_floor_clamp(self):
        ekf = ESKF(p0_std=0.001)
        gate = ekf.innovation_gate(floor=5.0)
        self.assertGreaterEqual(gate, 5.0)


class TestGravitySign(_Base):
    def test_constructor_validates_pm1(self):
        with self.assertRaises(ValueError):
            ESKF(gravity_sign=0.5)
        with self.assertRaises(ValueError):
            ESKF(gravity_sign=2.0)

    def test_negative_gravity_flips_g_world(self):
        ekf = ESKF(gravity_sign=-1.0)
        self.assertLess(ekf.g_world[2], 0)

    def test_positive_gravity_default(self):
        ekf = ESKF()
        self.assertGreater(ekf.g_world[2], 0)


class TestParameterValidation(_Base):
    def test_bad_thrust_rejected(self):
        with self.assertRaises(ValueError):
            ESKF(c_thrust=-1.0)
        with self.assertRaises(ValueError):
            ESKF(c_thrust=float("nan"))

    def test_bad_drag_rejected(self):
        with self.assertRaises(ValueError):
            ESKF(drag_kd=-0.1)
        with self.assertRaises(ValueError):
            ESKF(drag_kd=float("inf"))


class TestSignVariants(_Base):
    def test_both_conventions_hover_under_01m(self):
        dt = self.dt
        hover = _hover_thrust()
        rng = self.rng
        ekf_pos = ESKF(
            p0=np.zeros(3),
            v0=np.zeros(3),
            q0=self.q0,
            sigma_accel=2.0,
            gravity_sign=1.0,
        )
        ekf_neg = ESKF(
            p0=np.zeros(3),
            v0=np.zeros(3),
            q0=self.q0,
            sigma_accel=2.0,
            gravity_sign=-1.0,
        )
        for _ in range(500):
            ekf_pos.predict_commanded(hover, rng.normal(0, 0.002, 3), dt)
            ekf_neg.predict_commanded(hover, rng.normal(0, 0.002, 3), dt)
        d_pos = np.linalg.norm(ekf_pos.p)
        d_neg = np.linalg.norm(ekf_neg.p)
        self.assertLess(d_pos, 0.1, f"NED hover drift {d_pos:.6f}m >= 0.1m")
        self.assertLess(d_neg, 0.1, f"flipped hover drift {d_neg:.6f}m >= 0.1m")

    def test_matched_trajectory_within_10pct(self):
        dt = self.dt
        hover = _hover_thrust()
        rng = self.rng
        v_init = np.array([2.0, 1.0, -0.5])
        n = 500

        ekf_pos = ESKF(
            p0=np.zeros(3), v0=v_init, q0=self.q0, sigma_accel=2.0, gravity_sign=1.0
        )
        ekf_neg = ESKF(
            p0=np.zeros(3), v0=v_init, q0=self.q0, sigma_accel=2.0, gravity_sign=-1.0
        )
        for _ in range(n):
            g = rng.normal(0, 0.002, 3)
            ekf_pos.predict_commanded(hover, g, dt)
            ekf_neg.predict_commanded(hover, g, dt)

        d_pos = np.linalg.norm(ekf_pos.p)
        d_neg = np.linalg.norm(ekf_neg.p)
        ratio = abs(d_pos - d_neg) / max(d_pos, 1e-12)
        self.assertLess(
            ratio, 0.10, f"sign matched trajectory ratio {ratio:.4f} >= 0.10"
        )


class TestDropoutRecovery(_Base):
    def test_2s_dropout_then_vision(self):
        dt = self.dt
        T = 10.0
        n = int(T / dt)
        hover = _hover_thrust()
        v0 = np.array([1.0, 0.5, -0.3])
        kd = DRAG_KD
        ekf = ESKF(p0=np.zeros(3), v0=v0, q0=self.q0, sigma_accel=2.0)
        errs = []
        dropout_start = int(3.0 / dt)
        dropout_end = int(5.0 / dt)
        for k in range(n):
            t = k * dt
            gyro = self._clean_gyro()
            ekf.predict_commanded(hover, gyro, dt)
            pt = v0 / kd * (1.0 - math.exp(-kd * t))
            if k % 20 == 0 and not (dropout_start <= k < dropout_end):
                ekf.update_position(pt + self.rng.normal(0, 0.15, 3))
            if k == dropout_end:
                ekf.coast(int(0.5 / dt), dt)
            errs.append(np.linalg.norm(ekf.p - pt))
        final_err = float(np.mean(errs[-100:]))
        self.assertLess(final_err, 0.5, f"post-dropout pos err {final_err:.2f} >= 0.5m")


class TestZUPTDispatch(_Base):
    def test_zero_thrust_zeros_velocity(self):
        ekf = ESKF(p0=np.zeros(3), v0=np.array([3, 0, 0]), q0=self.q0, sigma_accel=2.0)
        for _ in range(200):
            ekf.predict_commanded(0.0, self._clean_gyro(), self.dt)
        assert_allclose(ekf.v, 0, atol=1e-15)

    def test_small_thrust_dispatches_zupt(self):
        ekf = ESKF(p0=np.zeros(3), v0=np.array([1, 0, 0]), q0=self.q0, sigma_accel=2.0)
        for _ in range(50):
            ekf.predict_commanded(0.04, self._clean_gyro(), self.dt)
        assert_allclose(ekf.v, 0, atol=1e-15)


class TestZUPTPropagation(_Base):
    def test_zupt_freezes_velocity(self):
        ekf = ESKF(p0=np.zeros(3), v0=np.array([3, 0, 0]), q0=self.q0)
        for _ in range(100):
            ekf.predict_zupt(self._clean_gyro(), self.dt)
        assert_allclose(ekf.v, 0, atol=1e-15)

    def test_zupt_gyro_spike_rejected(self):
        ekf = ESKF(q0=self.q0)
        q_before = ekf.q.copy()
        ekf.predict_zupt(np.array([0, 0, GYRO_SPIKE + 1]), self.dt)
        assert_allclose(ekf.q, q_before)


class TestPredictCommandedGuards(_Base):
    def test_bad_dt_rejected(self):
        ekf = ESKF(q0=self.q0)
        tr0 = ekf.state()["P_trace"]
        ekf.predict_commanded(0.5, np.zeros(3), 0.0)
        ekf.predict_commanded(0.5, np.zeros(3), -1.0)
        ekf.predict_commanded(0.5, np.zeros(3), 1.0)
        self.assertAlmostEqual(ekf.state()["P_trace"], tr0)

    def test_nan_gyro_rejected(self):
        ekf = ESKF(q0=self.q0)
        tr0 = ekf.state()["P_trace"]
        ekf.predict_commanded(0.5, np.array([float("nan"), 0, 0]), self.dt)
        self.assertAlmostEqual(ekf.state()["P_trace"], tr0)

    def test_nan_thrust_rejected(self):
        ekf = ESKF(q0=self.q0)
        tr0 = ekf.state()["P_trace"]
        ekf.predict_commanded(float("nan"), np.zeros(3), self.dt)
        self.assertAlmostEqual(ekf.state()["P_trace"], tr0)

    def test_wrong_shape_and_type_rejected(self):
        ekf = ESKF(q0=self.q0)
        state_before = ekf.state()
        ekf.predict_commanded(0.5, np.zeros(2), self.dt)
        ekf.predict_commanded("bad", np.zeros(3), self.dt)
        assert_allclose(ekf.p, state_before["p"])
        assert_allclose(ekf.v, state_before["v"])
        assert_allclose(ekf.q, state_before["q"])
        self.assertAlmostEqual(ekf.state()["P_trace"], state_before["P_trace"])


class TestPredictPreserved(_Base):
    def test_original_predict_signature_unchanged(self):
        ekf = ESKF(p0=np.array([1, 2, 3]), v0=np.zeros(3), q0=self.q0)
        accel = np.array([0, 0, -GRAVITY])
        gyro = np.zeros(3)
        p_before = ekf.p.copy()
        ekf.predict(accel, gyro, self.dt)
        assert_allclose(ekf.p[0], p_before[0], atol=1e-10)


class TestDragCovarianceRegression(_Base):
    def test_drag_reduces_velocity_uncertainty_growth(self):
        dt = self.dt
        hover = _hover_thrust()
        rng = self.rng
        n = 300
        ekf_drag = ESKF(q0=self.q0, sigma_accel=2.0, drag_kd=DRAG_KD)
        ekf_nodrag = ESKF(q0=self.q0, sigma_accel=2.0, drag_kd=0.0)
        for _ in range(n):
            g = rng.normal(0, 0.002, 3)
            ekf_drag.predict_commanded(hover, g, dt)
            ekf_nodrag.predict_commanded(hover, g, dt)
        v_drag = float(np.trace(ekf_drag.P[3:6, 3:6]))
        v_nodrag = float(np.trace(ekf_nodrag.P[3:6, 3:6]))
        self.assertLess(
            v_drag, v_nodrag, f"drag vel cov {v_drag:.3f} >= nodrag {v_nodrag:.3f}"
        )
        diff_d = ekf_drag.P - ekf_drag.P.T
        diff_n = ekf_nodrag.P - ekf_nodrag.P.T
        self.assertLess(np.max(np.abs(diff_d)), 1e-12, "drag cov not symmetric")
        self.assertLess(np.max(np.abs(diff_n)), 1e-12, "nodrag cov not symmetric")
        eigvals_d = np.linalg.eigvalsh(ekf_drag.P)
        self.assertTrue(np.all(eigvals_d >= -1e-12), "drag P not PSD")
        eigvals_n = np.linalg.eigvalsh(ekf_nodrag.P)
        self.assertTrue(np.all(eigvals_n >= -1e-12), "nodrag P not PSD")


class TestCovarianceSymmetry(_Base):
    def test_symmetry_after_mixed_operations(self):
        ekf = ESKF(q0=self.q0, sigma_accel=2.0)
        hover = _hover_thrust()
        for _ in range(100):
            ekf.predict_commanded(hover, self._clean_gyro(), self.dt)
        ekf.coast(20, self.dt)
        ekf.update_position(np.zeros(3), sigma=0.3)
        ekf.notify_collision()
        diff = ekf.P - ekf.P.T
        self.assertLess(np.max(np.abs(diff)), 1e-12)


class TestFusePnpGateProduction(_Base):
    def test_normal_accept(self):
        ekf = ESKF(p0=np.array([9.5, 0, -7.5]), q0=self.q0, p0_std=0.5)
        gate_world = _gate_world_at(10, 0, -3)
        gate_body = np.array([0, 0, 5])
        det = _good_det(gate_body)
        result = fuse_pnp_gate(ekf, det, gate_world)
        self.assertTrue(result)
        expected_pos = gate_world - quat_to_R(self.q0) @ gate_body
        self.assertLess(np.linalg.norm(ekf.p - expected_pos), 1.0)

    def test_gross_outlier_rejected(self):
        ekf = ESKF(p0=np.zeros(3), q0=self.q0, p0_std=0.5)
        gate_world = _gate_world_at(100, 0, -3)
        gate_body = np.array([0, 0, 5])
        det = _good_det(gate_body)
        self.assertFalse(fuse_pnp_gate(ekf, det, gate_world))

    def test_low_confidence_rejected(self):
        ekf = ESKF(p0=np.array([5, 0, -3]), q0=self.q0)
        gate_world = _gate_world_at(10, 0, -3)
        gate_body = np.array([0, 0, 5])
        self.assertFalse(fuse_pnp_gate(ekf, _good_det(gate_body, conf=0.3), gate_world))

    def test_empty_det_rejected(self):
        ekf = ESKF(p0=np.array([5, 0, -3]), q0=self.q0)
        self.assertFalse(fuse_pnp_gate(ekf, {}, _gate_world_at(10, 0, -3)))

    def test_adaptive_gate_accepts_after_coast(self):
        ekf = ESKF(p0=np.array([4.5, 0, -2.5]), q0=self.q0, p0_std=2.0)
        ekf.coast(500, self.dt)
        gate_world = _gate_world_at(10, 0, -3)
        gate_body = np.array([0, 0, 5])
        det = _good_det(gate_body)
        p_meas = gate_world - quat_to_R(self.q0) @ gate_body
        innov = float(np.linalg.norm(p_meas - ekf.p))
        gate = ekf.innovation_gate(6.0)
        self.assertGreater(
            gate, innov, f"gate {gate:.2f} should exceed innov {innov:.2f}"
        )
        self.assertTrue(fuse_pnp_gate(ekf, det, gate_world))

    def test_nonfinite_gate_body_rejected(self):
        ekf = ESKF(p0=np.array([9.5, 0, -7.5]), q0=self.q0, p0_std=0.5)
        det = _good_det(np.array([float("nan"), 0, 0]))
        self.assertFalse(fuse_pnp_gate(ekf, det, _gate_world_at(10, 0, -3)))

    def test_nonfinite_gate_world_rejected(self):
        ekf = ESKF(p0=np.array([9.5, 0, -7.5]), q0=self.q0, p0_std=0.5)
        det = _good_det(np.array([0, 0, 5]))
        self.assertFalse(fuse_pnp_gate(ekf, det, np.array([float("inf"), 0, 0])))

    def test_malformed_scalar_fields_rejected(self):
        ekf = ESKF(p0=np.array([9.5, 0, -7.5]), q0=self.q0, p0_std=0.5)
        gate_world = _gate_world_at(10, 0, -3)
        for det in (
            {"conf": "bad", "pose": {"gate_pos_body": [0, 0, 5]}},
            {"conf": 0.9, "pose": {"gate_pos_body": [0, 0, 5], "reproj_px": object()}},
            {"conf": 0.9, "pose": "bad"},
        ):
            with self.subTest(det=det):
                self.assertFalse(fuse_pnp_gate(ekf, det, gate_world))


class TestStepPnpFusion(_Base):
    def test_miss_coasts(self):
        ekf = ESKF(q0=self.q0)
        status = step_pnp_fusion(
            ekf, None, _gate_world_at(10, 0, -3), self.dt, max_coast=100
        )
        self.assertEqual(status, "miss")
        self.assertEqual(ekf.coast_count, 1)

    def test_normal_update(self):
        ekf = ESKF(p0=np.array([9.5, 0, -7.5]), q0=self.q0, p0_std=0.5)
        gate_world = _gate_world_at(10, 0, -3)
        det = _good_det(np.array([0, 0, 5]))
        status = step_pnp_fusion(ekf, det, gate_world, self.dt, max_coast=100)
        self.assertEqual(status, "update")
        self.assertEqual(ekf.coast_count, 0)

    def test_rejected_outlier(self):
        ekf = ESKF(p0=np.zeros(3), q0=self.q0, p0_std=0.1)
        det = _good_det(np.array([0, 0, 5]))
        status = step_pnp_fusion(
            ekf, det, _gate_world_at(100, 0, -3), self.dt, max_coast=100
        )
        self.assertEqual(status, "rejected")
        self.assertGreater(ekf.coast_count, 0)

    def test_prolonged_loss_hard_reset(self):
        ekf = ESKF(p0=np.array([5, 0, -3]), q0=self.q0)
        gate_world = _gate_world_at(10, 0, -3)
        gate_body = np.array([0, 0, 5])
        max_coast = 5
        for _ in range(max_coast):
            step_pnp_fusion(ekf, None, gate_world, self.dt, max_coast=max_coast)
        self.assertEqual(ekf.coast_count, max_coast)
        status = step_pnp_fusion(
            ekf, _good_det(gate_body), gate_world, self.dt, max_coast=max_coast
        )
        self.assertEqual(status, "reset")
        self.assertEqual(ekf.coast_count, 0)
        self.assertGreater(ekf.recovery_count, 0)

    def test_moderate_uncertain_recovery(self):
        ekf = ESKF(p0=np.array([4, 0, -2]), q0=self.q0, p0_std=2.0)
        ekf.coast(200, self.dt)
        gate_world = _gate_world_at(10, 0, -3)
        gate_body = np.array([0, 0, 5])
        det = _good_det(gate_body)
        status = step_pnp_fusion(ekf, det, gate_world, self.dt, max_coast=1000)
        self.assertEqual(status, "update")
        expected_pos = gate_world - quat_to_R(self.q0) @ gate_body
        err = float(np.linalg.norm(ekf.p - expected_pos))
        self.assertLess(err, 1.0, f"recovery err {err:.2f}m too large")

    def test_unhealthy_bad_q_rejects(self):
        ekf = ESKF(q0=self.q0)
        ekf.P[0, 0] = float("inf")
        ekf.q = np.array([1, float("nan"), 0, 0])
        gate_world = _gate_world_at(10, 0, -3)
        gate_body = np.array([0, 0, 5])
        det = _good_det(gate_body)
        status = step_pnp_fusion(ekf, det, gate_world, self.dt, max_coast=1000)
        self.assertEqual(status, "rejected")

    def test_unhealthy_good_q_resets(self):
        ekf = ESKF(q0=self.q0)
        ekf.P[0, 0] = float("inf")
        ekf.p[0] = float("nan")
        gate_world = _gate_world_at(10, 0, -3)
        gate_body = np.array([0, 0, 5])
        det = _good_det(gate_body)
        status = step_pnp_fusion(ekf, det, gate_world, self.dt, max_coast=1000)
        self.assertEqual(status, "reset")
        self.assertTrue(ekf.healthy())

    def test_nonfinite_gate_world_missions(self):
        ekf = ESKF(q0=self.q0, p0_std=0.5)
        status = step_pnp_fusion(
            ekf, None, np.array([float("inf"), 0, 0]), self.dt, max_coast=10
        )
        self.assertEqual(status, "miss")

    def test_invalid_pose_no_state_mutation(self):
        ekf = ESKF(p0=np.array([5, 0, -3]), q0=self.q0, p0_std=0.5)
        p_before = ekf.p.copy()
        v_before = ekf.v.copy()
        bad_det = _good_det(np.array([float("nan"), 0, 0]))
        status = step_pnp_fusion(
            ekf, bad_det, _gate_world_at(10, 0, -3), self.dt, max_coast=10
        )
        self.assertEqual(status, "miss")
        assert_allclose(ekf.p, p_before)
        assert_allclose(ekf.v, v_before)


if __name__ == "__main__":
    unittest.main()
