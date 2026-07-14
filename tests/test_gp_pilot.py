"""Unit tests for AndurilGP controls port (AHRS, guidance, wiring)."""

import math
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from simulator.gp_pilot import (
    HOVER_THRUST,
    K_BEARING,
    MAX_BANK_DEG,
    PERP_BLEND_DIST,
    _fresh_hold_state,
    compute_guidance,
)
from simulator.gp_vision import gate_body_from_pinhole, vision_gate_estimate
from simulator.gyro_ahrs import GyroAHRS, euler_to_quat


class GyroAHRSTests(unittest.TestCase):
    def test_seed_pitch(self):
        ahrs = GyroAHRS(initial_pitch_deg=-17.8)
        r, p, y = ahrs.euler_deg()
        self.assertAlmostEqual(r, 0.0, places=5)
        self.assertAlmostEqual(p, -17.8, places=4)
        self.assertAlmostEqual(y, 0.0, places=5)

    def test_integrate_yaw_rate(self):
        ahrs = GyroAHRS(initial_pitch_deg=0.0)
        # Integrate +1 rad/s yaw for 0.5 s → ~28.6 deg
        for _ in range(50):
            ahrs.update(0.0, 0.0, 1.0, 0.01)
        _r, _p, y = ahrs.euler_deg()
        self.assertAlmostEqual(y, math.degrees(0.5), places=1)

    def test_quat_roundtrip(self):
        q = euler_to_quat(0.1, -0.2, 0.3)
        self.assertAlmostEqual(sum(x * x for x in q), 1.0, places=6)


class GuidanceTests(unittest.TestCase):
    def _level_quat(self):
        return np.array(euler_to_quat(0.0, 0.0, 0.0), dtype=np.float64)

    def test_blend_ramp(self):
        # far: bx=12 → blend=1; near: bx=0 → blend=0; mid bx=3 → 0.5
        self.assertAlmostEqual(float(np.clip(12.0 / PERP_BLEND_DIST, 0, 1)), 1.0)
        self.assertAlmostEqual(float(np.clip(0.0 / PERP_BLEND_DIST, 0, 1)), 0.0)
        self.assertAlmostEqual(float(np.clip(3.0 / PERP_BLEND_DIST, 0, 1)), 0.5)

    def test_bearing_bank_far_gate(self):
        state = _fresh_hold_state()
        # Gate 12 m ahead, 0.8 m right → ~3.8 deg bearing, full blend, under clip
        vision = {
            "frame_id": 1,
            "body_x_m": 12.0,
            "body_y_m": 0.8,
            "body_z_m": 0.0,
            "normal_body": None,
        }
        _rr, _pr, _yr, thrust, dbg = compute_guidance(
            roll_deg=0.0,
            pitch_deg=0.0,
            quat=self._level_quat(),
            vY=0.0,
            vD=0.0,
            vision=vision,
            vision_vel=None,
            state=state,
        )
        self.assertTrue(dbg["vision_valid"])
        self.assertAlmostEqual(dbg["blend"], 1.0, places=5)
        expected_bearing = math.degrees(math.atan2(0.8, 12.0))
        self.assertAlmostEqual(dbg["bearing_deg"], expected_bearing, places=4)
        self.assertAlmostEqual(
            dbg["desired_roll"], K_BEARING * expected_bearing, places=3
        )
        self.assertLessEqual(abs(dbg["desired_roll"]), MAX_BANK_DEG)
        self.assertGreater(thrust, 0.0)

    def test_elev_thrust_gate_below(self):
        state = _fresh_hold_state()
        # Gate below drone in body: large +bz at level → positive elev_err (NED down)
        # With identity quat, gate_pD = bz
        vision = {
            "frame_id": 5,
            "body_x_m": 10.0,
            "body_y_m": 0.0,
            "body_z_m": 2.0,  # below
            "normal_body": None,
        }
        _rr, _pr, _yr, thrust, dbg = compute_guidance(
            roll_deg=0.0,
            pitch_deg=0.0,
            quat=self._level_quat(),
            vY=0.0,
            vD=0.0,
            vision=vision,
            vision_vel=None,
            state=state,
        )
        self.assertAlmostEqual(dbg["elev_err"], 2.0, places=4)
        # Positive elev → reduce thrust below hover
        self.assertLess(thrust, HOVER_THRUST)

    def test_blend_zero_at_gate_plane(self):
        state = _fresh_hold_state()
        vision = {
            "frame_id": 2,
            "body_x_m": 0.15,  # valid (>0.1) but blend≈0
            "body_y_m": 1.0,
            "body_z_m": 0.0,
            "normal_body": None,
        }
        _rr, _pr, _yr, _t, dbg = compute_guidance(
            roll_deg=0.0,
            pitch_deg=0.0,
            quat=self._level_quat(),
            vY=0.0,
            vD=0.0,
            vision=vision,
            vision_vel=None,
            state=state,
        )
        self.assertLess(dbg["blend"], 0.03)
        # Bank authority scales with blend → near-gate roll command is small
        self.assertLess(abs(dbg["desired_roll"]), 4.0)

    def test_no_vision_hoverish(self):
        state = _fresh_hold_state()
        _rr, _pr, _yr, thrust, dbg = compute_guidance(
            roll_deg=0.0,
            pitch_deg=-3.0,
            quat=self._level_quat(),
            vY=0.0,
            vD=0.0,
            vision=None,
            vision_vel=None,
            state=state,
        )
        self.assertFalse(dbg["vision_valid"])
        self.assertAlmostEqual(thrust, HOVER_THRUST, places=3)


class VisionAdapterTests(unittest.TestCase):
    def test_pinhole_body_ahead(self):
        body = gate_body_from_pinhole(320.0, 180.0, box_w_px=90.0)
        self.assertIsNotNone(body)
        bx, by, bz = body
        self.assertGreater(bx, 0.0)
        self.assertAlmostEqual(by, 0.0, places=3)

    def test_vision_from_pose_pkt(self):
        data = {
            "pose": {
                "frame_id": 7,
                "gates": [
                    {
                        "conf": 0.9,
                        "pose": {
                            "gate_pos_body": np.array([8.0, -0.5, 0.2]),
                            "normal_body": np.array([-1.0, 0.0, 0.0]),
                        },
                    }
                ],
            }
        }
        est = vision_gate_estimate(data)
        self.assertIsNotNone(est)
        self.assertEqual(est["frame_id"], 7)
        self.assertAlmostEqual(est["body_x_m"], 8.0)
        self.assertTrue(est["pnp_ok"])


class WiringTests(unittest.TestCase):
    def test_controller_uses_gp_pilot(self):
        with patch.dict("os.environ", {"AUTO_PILOT": "gp"}, clear=False):
            from simulator.controller import Controller
            from simulator.gp_pilot import GPPilot

            ctrl = Controller(MagicMock(), {}, 0)
            self.assertIsInstance(ctrl.pilot, GPPilot)
            ctrl.pilot.shutdown()

    def test_default_auto_still_ibvs(self):
        with patch.dict(
            "os.environ", {"AUTO_FLIGHT": "1", "AUTO_PILOT": "ibvs"}, clear=False
        ):
            from simulator.controller import Controller
            from simulator.ibvs_pilot import IBVSPilot

            ctrl = Controller(MagicMock(), {}, 0)
            self.assertIsInstance(ctrl.pilot, IBVSPilot)

    def test_gp_waits_before_flying(self):
        from simulator.gp_pilot import GPPilot, Phase

        ctrl = MagicMock()
        pilot = GPPilot(ctrl, {"armed": False})
        self.assertEqual(pilot.phase, Phase.WAIT_FOR_DATA)
        pilot.tick()
        args = ctrl.set_attitude_rates.call_args[0]
        self.assertEqual(args, (0.0, 0.0, 0.0, 0.0))
        pilot.shutdown()


if __name__ == "__main__":
    unittest.main()
