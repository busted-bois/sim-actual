"""Focused tests for the three ported PR #21 guidance equations.

1. Gate-normal doffset approach
2. Cross-track error bank correction
3. Anduril optical-elevation correction
The complementary AHRS experiment is intentionally excluded: it regressed
the existing closed-loop attitude contract.
"""

import math
import unittest

import numpy as np

from simulator.anduril_gate_detect import CAM_TILT_DEG as CANONICAL_CAM_TILT
from simulator.gp_pilot import (
    CENTERED_BY_M,
    DOFFSET_FADE_FAR_M,
    DOFFSET_FADE_NEAR_M,
    DOFFSET_M,
    E_SIGNED_CLIP_M,
    K_CROSS,
    _fresh_hold_state,
    apply_normal_approach_body,
    compute_guidance,
    cross_track_error,
    gate_through_ned,
    through_dhat_body,
)
from simulator.gyro_ahrs import euler_to_quat


def _level_quat():
    return np.array(euler_to_quat(0.0, 0.0, 0.0), dtype=np.float64)


def _yaw90_quat():
    return np.array(euler_to_quat(0.0, 0.0, math.pi / 2), dtype=np.float64)


class CanonicalTiltTest(unittest.TestCase):
    def test_imported_from_anduril_gate_detect(self):
        self.assertEqual(CANONICAL_CAM_TILT, 20.0)


class GateThroughNedTests(unittest.TestCase):
    def test_identity_quat(self):
        n = gate_through_ned(_level_quat())
        self.assertIsNotNone(n)
        np.testing.assert_allclose(n, [1.0, 0.0, 0.0], atol=1e-6)

    def test_none_input(self):
        self.assertIsNone(gate_through_ned(None))

    def test_east_gate_yaw90(self):
        n = gate_through_ned(_yaw90_quat())
        self.assertIsNotNone(n)
        np.testing.assert_allclose(n, [0.0, 1.0, 0.0], atol=1e-6)

    def test_unit_norm(self):
        for q in [_level_quat(), _yaw90_quat(), np.array(euler_to_quat(0, 0, 0.45))]:
            n = gate_through_ned(q)
            if n is not None:
                self.assertAlmostEqual(float(np.linalg.norm(n)), 1.0, places=6)

    def test_degenerate_zero_quat(self):
        self.assertIsNone(gate_through_ned(np.array([0.0, 0.0, 0.0, 0.0])))

    def test_nonfinite_rejected(self):
        self.assertIsNone(gate_through_ned(np.array([float("inf"), 0, 0, 0])))

    def test_malformed_shape_rejected(self):
        self.assertIsNone(gate_through_ned([1.0, 0.0, 0.0]))


class ThroughDhatBodyTests(unittest.TestCase):
    def test_identity_body_from_map(self):
        d = through_dhat_body(_level_quat(), _level_quat())
        np.testing.assert_allclose(d, [1.0, 0.0, 0.0], atol=1e-9)

    def test_east_gate_yaw90_drone_level(self):
        d = through_dhat_body(_level_quat(), _yaw90_quat())
        np.testing.assert_allclose(d, [0.0, 1.0, 0.0], atol=1e-9)

    def test_east_gate_yaw90_drone_yaw90_still_body_x(self):
        d = through_dhat_body(_yaw90_quat(), _yaw90_quat())
        np.testing.assert_allclose(d, [1.0, 0.0, 0.0], atol=1e-9)

    def test_pnp_normal_fallback(self):
        d = through_dhat_body(_level_quat(), normal_body=np.array([-1.0, 0.0, 0.0]))
        np.testing.assert_allclose(d, [1.0, 0.0, 0.0], atol=1e-9)

    def test_none_when_no_data(self):
        self.assertIsNone(through_dhat_body(_level_quat(), None, None))

    def test_pnp_normal_zero_rejected(self):
        self.assertIsNone(
            through_dhat_body(_level_quat(), None, np.array([0.0, 0.0, 0.0]))
        )

    def test_pnp_normal_nan_rejected(self):
        self.assertIsNone(
            through_dhat_body(_level_quat(), None, np.array([float("nan"), 0.0, 0.0]))
        )

    def test_pnp_normal_malformed_rejected(self):
        self.assertIsNone(through_dhat_body(_level_quat(), None, [1.0, 0.0]))


class CrossTrackErrorTests(unittest.TestCase):
    def test_on_path_zero(self):
        dhat = np.array([1.0, 0.0, 0.0])
        e = cross_track_error(10.0, 0.0, 0.0, dhat)
        self.assertAlmostEqual(e, 0.0, places=6)

    def test_positive_by_negative_cte(self):
        dhat = np.array([1.0, 0.0, 0.0])
        e = cross_track_error(10.0, 2.0, 0.0, dhat)
        self.assertAlmostEqual(e, -2.0, places=6)

    def test_negative_by_positive_cte(self):
        dhat = np.array([1.0, 0.0, 0.0])
        e = cross_track_error(10.0, -1.5, 0.0, dhat)
        self.assertAlmostEqual(e, 1.5, places=6)

    def test_clipped(self):
        dhat = np.array([1.0, 0.0, 0.0])
        e = cross_track_error(10.0, 99.0, 0.0, dhat)
        self.assertAlmostEqual(e, -E_SIGNED_CLIP_M, places=6)

    def test_returns_scalar_float(self):
        dhat = np.array([1.0, 0.0, 0.0])
        e = cross_track_error(5.0, 1.0, 0.0, dhat)
        self.assertIsInstance(e, float)
        self.assertNotIsInstance(e, tuple)


class NormalApproachTests(unittest.TestCase):
    def test_doffset_zero_is_noop(self):
        ax, ay, az, used = apply_normal_approach_body(
            12.0, 0.0, 0.0, np.array([1, 0, 0.0]), 0.0
        )
        self.assertEqual((ax, ay, az, used), (12.0, 0.0, 0.0, 0.0))

    def test_head_on_shortens_bx(self):
        d = np.array([1.0, 0.0, 0.0])
        ax, ay, az, used = apply_normal_approach_body(12.0, 0.0, 0.0, d, DOFFSET_M)
        self.assertAlmostEqual(used, DOFFSET_M)
        self.assertAlmostEqual(ax, 12.0 - DOFFSET_M)
        self.assertAlmostEqual(ay, 0.0)

    def test_none_through_noop(self):
        ax, ay, az, used = apply_normal_approach_body(12.0, 0.0, 0.0, None, 1.5)
        self.assertEqual((ax, ay, az, used), (12.0, 0.0, 0.0, 0.0))

    def test_bx_too_close_noop(self):
        d = np.array([1.0, 0.0, 0.0])
        ax, ay, az, used = apply_normal_approach_body(0.5, 0.0, 0.0, d, 1.5)
        self.assertEqual((ax, ay, az, used), (0.5, 0.0, 0.0, 0.0))

    def test_angled_gate_shifts_all_components(self):
        d = np.array([0.707, 0.707, 0.0])
        ax, ay, az, used = apply_normal_approach_body(10.0, 0.0, 0.0, d, 2.0)
        self.assertAlmostEqual(used, 2.0)
        self.assertAlmostEqual(ax, 10.0 - 2.0 * 0.707, places=4)
        self.assertAlmostEqual(ay, 0.0 - 2.0 * 0.707, places=4)

    def test_vertical_gate_shifts_az(self):
        d = np.array([0.0, 0.0, 1.0])
        ax, ay, az, used = apply_normal_approach_body(10.0, 0.0, -1.0, d, 1.0)
        self.assertAlmostEqual(used, 1.0)
        self.assertAlmostEqual(ax, 10.0, places=4)
        self.assertAlmostEqual(az, -1.0 - 1.0, places=4)

    def test_nonfinite_through_noop(self):
        d = np.array([float("nan"), 0.0, 0.0])
        ax, ay, az, used = apply_normal_approach_body(12.0, 0.0, 0.0, d, 1.0)
        self.assertEqual((ax, ay, az, used), (12.0, 0.0, 0.0, 0.0))

    def test_malformed_through_noop(self):
        result = apply_normal_approach_body(12.0, 0.0, 0.0, [1.0, 0.0], 1.0)
        self.assertEqual(result, (12.0, 0.0, 0.0, 0.0))


class DoffsetGuidanceTests(unittest.TestCase):
    def test_doffset_m_zero_inert(self):
        vision = {
            "frame_id": 1,
            "body_x_m": 15.0,
            "body_y_m": 0.0,
            "body_z_m": 0.0,
        }
        _a, _b, _c, _t, dbg = compute_guidance(
            roll_deg=0.0,
            pitch_deg=0.0,
            quat=_level_quat(),
            vY=0.0,
            vD=0.0,
            vision=vision,
            vision_vel=None,
            state=_fresh_hold_state(),
            gate_quat=_level_quat(),
            doffset_m=0.0,
        )
        self.assertEqual(dbg["doffset"], 0.0)
        self.assertAlmostEqual(dbg["bx"], 15.0, places=4)

    def test_full_doffset_far(self):
        vision = {
            "frame_id": 1,
            "body_x_m": DOFFSET_FADE_FAR_M + 1.0,
            "body_y_m": 0.3,
            "body_z_m": 0.0,
        }
        _a, _b, _c, _t, dbg = compute_guidance(
            roll_deg=0.0,
            pitch_deg=0.0,
            quat=_level_quat(),
            vY=0.0,
            vD=0.0,
            vision=vision,
            vision_vel=None,
            state=_fresh_hold_state(),
            gate_quat=_level_quat(),
            doffset_m=DOFFSET_M,
        )
        self.assertAlmostEqual(dbg["doffset"], DOFFSET_M, places=3)
        self.assertAlmostEqual(
            dbg["bx"], DOFFSET_FADE_FAR_M + 1.0 - DOFFSET_M, places=3
        )

    def test_fade_near_throat(self):
        vision = {
            "frame_id": 1,
            "body_x_m": DOFFSET_FADE_NEAR_M,
            "body_y_m": 0.0,
            "body_z_m": 0.0,
        }
        _a, _b, _c, _t, dbg = compute_guidance(
            roll_deg=0.0,
            pitch_deg=0.0,
            quat=_level_quat(),
            vY=0.0,
            vD=0.0,
            vision=vision,
            vision_vel=None,
            state=_fresh_hold_state(),
            gate_quat=_level_quat(),
            doffset_m=DOFFSET_M,
        )
        self.assertEqual(dbg["doffset"], 0.0)
        self.assertAlmostEqual(dbg["bx"], DOFFSET_FADE_NEAR_M, places=4)

    def test_centered_hole_keeps_raw_by(self):
        vision = {
            "frame_id": 1,
            "body_x_m": 15.0,
            "body_y_m": CENTERED_BY_M * 0.5,
            "body_z_m": 0.0,
        }
        _a, _b, _c, _t, dbg = compute_guidance(
            roll_deg=0.0,
            pitch_deg=0.0,
            quat=_level_quat(),
            vY=0.0,
            vD=0.0,
            vision=vision,
            vision_vel=None,
            state=_fresh_hold_state(),
            gate_quat=_level_quat(),
            doffset_m=DOFFSET_M,
        )
        self.assertAlmostEqual(dbg["by"], CENTERED_BY_M * 0.5, places=4)


class CTEGuidanceTests(unittest.TestCase):
    def test_cte_adds_bank_bias(self):
        vision = {
            "frame_id": 1,
            "body_x_m": 12.0,
            "body_y_m": 0.4,
            "body_z_m": 0.0,
        }
        _a, _b, _c, _t, dbg0 = compute_guidance(
            roll_deg=0.0,
            pitch_deg=0.0,
            quat=_level_quat(),
            vY=0.0,
            vD=0.0,
            vision=vision,
            vision_vel=None,
            state=_fresh_hold_state(),
            gate_quat=None,
            doffset_m=0.0,
        )
        _a, _b, _c, _t, dbg1 = compute_guidance(
            roll_deg=0.0,
            pitch_deg=0.0,
            quat=_level_quat(),
            vY=0.0,
            vD=0.0,
            vision=vision,
            vision_vel=None,
            state=_fresh_hold_state(),
            gate_quat=_level_quat(),
            doffset_m=0.0,
        )
        self.assertAlmostEqual(dbg1["e_signed"], -0.4, places=4)
        self.assertEqual(dbg0["e_signed"], 0.0)
        self.assertGreater(dbg1["desired_roll"], dbg0["desired_roll"])
        self.assertAlmostEqual(
            dbg1["desired_roll"] - dbg0["desired_roll"],
            K_CROSS * 0.4,
            places=3,
        )

    def test_negative_by_opposite_bank(self):
        vision = {
            "frame_id": 1,
            "body_x_m": 12.0,
            "body_y_m": -0.5,
            "body_z_m": 0.0,
        }
        _a, _b, _c, _t, dbg = compute_guidance(
            roll_deg=0.0,
            pitch_deg=0.0,
            quat=_level_quat(),
            vY=0.0,
            vD=0.0,
            vision=vision,
            vision_vel=None,
            state=_fresh_hold_state(),
            gate_quat=_level_quat(),
        )
        self.assertAlmostEqual(dbg["e_signed"], 0.5, places=4)
        self.assertLess(dbg["desired_roll"], 0.0)

    def test_cte_with_east_gate_and_rotated_drone(self):
        gq = _yaw90_quat()
        dq = _level_quat()
        # East gate: through-axis is [0,1,0]. CTE = bx*1 - by*0 = bx.
        # Zero by and non-zero bx -> nonzero CTE via dhat_y component.
        vision = {
            "frame_id": 1,
            "body_x_m": 0.0,
            "body_y_m": 0.0,
            "body_z_m": 0.0,
        }
        _a, _b, _c, _t, dbg = compute_guidance(
            roll_deg=0.0,
            pitch_deg=0.0,
            quat=dq,
            vY=0.0,
            vD=0.0,
            vision=vision,
            vision_vel=None,
            state=_fresh_hold_state(),
            gate_quat=gq,
            doffset_m=0.0,
        )
        # bx=0, by=0 -> CTE=0 regardless of through-axis
        self.assertEqual(dbg["e_signed"], 0.0)
        vision["body_y_m"] = 2.0
        _a2, _b2, _c2, _t2, dbg2 = compute_guidance(
            roll_deg=0.0,
            pitch_deg=0.0,
            quat=dq,
            vY=0.0,
            vD=0.0,
            vision=vision,
            vision_vel=None,
            state=_fresh_hold_state(),
            gate_quat=gq,
            doffset_m=0.0,
        )
        # through=[0,1,0], bx=0, by=2 -> CTE = 0*1 - 2*0 = 0
        self.assertAlmostEqual(dbg2["e_signed"], 0.0, places=4)
        vision2 = {
            "frame_id": 1,
            "body_x_m": 3.0,
            "body_y_m": 0.0,
            "body_z_m": 0.0,
        }
        _a3, _b3, _c3, _t3, dbg3 = compute_guidance(
            roll_deg=0.0,
            pitch_deg=0.0,
            quat=dq,
            vY=0.0,
            vD=0.0,
            vision=vision2,
            vision_vel=None,
            state=_fresh_hold_state(),
            gate_quat=gq,
            doffset_m=0.0,
        )
        # through=[0,1,0], bx=3, by=0 -> CTE = 3*1 - 0*0 = 3 (clipped to +2)
        self.assertAlmostEqual(dbg3["e_signed"], E_SIGNED_CLIP_M, places=4)


class OpticalElevTests(unittest.TestCase):
    def test_anduril_corrects_with_canonical_tilt(self):
        tilt_offset = math.tan(math.radians(CANONICAL_CAM_TILT)) * 10.0
        vision = {
            "frame_id": 1,
            "body_x_m": 10.0,
            "body_y_m": 0.0,
            "body_z_m": 0.0,
            "source": "anduril",
        }
        _a, _b, _c, _t, dbg = compute_guidance(
            roll_deg=0.0,
            pitch_deg=0.0,
            quat=_level_quat(),
            vY=0.0,
            vD=0.0,
            vision=vision,
            vision_vel=None,
            state=_fresh_hold_state(),
        )
        self.assertAlmostEqual(dbg["elev_err"], -tilt_offset, places=3)

    def test_non_anduril_no_correction(self):
        vision = {
            "frame_id": 1,
            "body_x_m": 10.0,
            "body_y_m": 0.0,
            "body_z_m": 0.0,
            "source": "pnp",
        }
        _a, _b, _c, _t, dbg = compute_guidance(
            roll_deg=0.0,
            pitch_deg=0.0,
            quat=_level_quat(),
            vY=0.0,
            vD=0.0,
            vision=vision,
            vision_vel=None,
            state=_fresh_hold_state(),
        )
        self.assertAlmostEqual(dbg["elev_err"], 0.0, places=5)

    def test_no_source_no_correction(self):
        vision = {
            "frame_id": 1,
            "body_x_m": 10.0,
            "body_y_m": 0.0,
            "body_z_m": 0.0,
        }
        _a, _b, _c, _t, dbg = compute_guidance(
            roll_deg=0.0,
            pitch_deg=0.0,
            quat=_level_quat(),
            vY=0.0,
            vD=0.0,
            vision=vision,
            vision_vel=None,
            state=_fresh_hold_state(),
        )
        self.assertAlmostEqual(dbg["elev_err"], 0.0, places=5)


class BackwardCompatTests(unittest.TestCase):
    def test_guidance_without_gate_context(self):
        vision = {
            "frame_id": 1,
            "body_x_m": 8.0,
            "body_y_m": 0.5,
            "body_z_m": 0.0,
        }
        _a, _b, _c, _t, dbg = compute_guidance(
            roll_deg=0.0,
            pitch_deg=0.0,
            quat=_level_quat(),
            vY=0.0,
            vD=0.0,
            vision=vision,
            vision_vel=None,
            state=_fresh_hold_state(),
        )
        self.assertEqual(dbg["doffset"], 0.0)
        self.assertEqual(dbg["e_signed"], 0.0)
        expected = math.degrees(math.atan2(0.5, 8.0))
        self.assertAlmostEqual(dbg["bearing_deg"], expected, places=4)

    def test_no_vision_hover(self):
        _a, _b, _c, thrust, dbg = compute_guidance(
            roll_deg=0.0,
            pitch_deg=0.0,
            quat=_level_quat(),
            vY=0.0,
            vD=0.0,
            vision=None,
            vision_vel=None,
            state=_fresh_hold_state(),
        )
        self.assertFalse(dbg["vision_valid"])

    def test_nonfinite_vision_is_rejected(self):
        vision = {
            "frame_id": 1,
            "body_x_m": float("inf"),
            "body_y_m": 0.0,
            "body_z_m": 0.0,
        }
        roll, pitch, yaw, thrust, dbg = compute_guidance(
            roll_deg=0.0,
            pitch_deg=0.0,
            quat=_level_quat(),
            vY=0.0,
            vD=0.0,
            vision=vision,
            vision_vel=None,
            state=_fresh_hold_state(),
        )
        self.assertFalse(dbg["vision_valid"])
        self.assertTrue(all(math.isfinite(v) for v in (roll, pitch, yaw, thrust)))


if __name__ == "__main__":
    unittest.main()
