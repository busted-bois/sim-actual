"""Unit tests for AndurilGP controls port (AHRS, guidance, wiring)."""

import math
import time
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

    def test_speed_loop_pitches_down_when_too_slow(self):
        from simulator.gp_pilot import CRUISE_SPEED_MPS, DESIRED_PITCH_DEG

        state = _fresh_hold_state()
        vision = {
            "frame_id": 10,
            "body_x_m": 12.0,
            "body_y_m": 0.0,
            "body_z_m": 0.0,
            "normal_body": None,
            "reliable": True,
        }
        # Trusted vX (above untrusted floor) but below cruise → nose down.
        _rr, _pr, _yr, _t, dbg = compute_guidance(
            roll_deg=0.0,
            pitch_deg=0.0,
            quat=self._level_quat(),
            vY=0.0,
            vD=0.0,
            vision=vision,
            vision_vel=None,
            state=state,
            vX=1.0,
        )
        self.assertAlmostEqual(dbg["v_target"], CRUISE_SPEED_MPS, places=2)
        self.assertLess(dbg["pitch_des_deg"], DESIRED_PITCH_DEG)

    def test_speed_loop_slows_near_gate(self):
        from simulator.gp_pilot import THRU_SPEED_MPS

        state = _fresh_hold_state()
        vision = {
            "frame_id": 11,
            "body_x_m": 1.0,
            "body_y_m": 0.0,
            "body_z_m": 0.0,
            "normal_body": None,
            "reliable": True,
        }
        _rr, _pr, _yr, _t, dbg = compute_guidance(
            roll_deg=0.0,
            pitch_deg=-3.0,
            quat=self._level_quat(),
            vY=0.0,
            vD=0.0,
            vision=vision,
            vision_vel=None,
            state=state,
            vX=2.2,
        )
        self.assertLess(dbg["v_target"], 1.6)
        self.assertGreaterEqual(dbg["v_target"], THRU_SPEED_MPS - 0.05)

    def test_hard_brake_when_over_max_speed(self):
        from simulator.gp_pilot import MAX_SPEED_MPS, PITCH_DES_MAX_DEG

        state = _fresh_hold_state()
        _rr, _pr, _yr, _t, dbg = compute_guidance(
            roll_deg=0.0,
            pitch_deg=0.0,
            quat=self._level_quat(),
            vY=0.0,
            vD=0.0,
            vision=None,
            vision_vel=None,
            state=state,
            vX=MAX_SPEED_MPS * 1.2,  # ~12 km/h
        )
        self.assertAlmostEqual(dbg["pitch_des_deg"], PITCH_DES_MAX_DEG, places=3)

    def test_no_vision_still_regulates_crawl(self):
        from simulator.gp_pilot import BLIND_CRAWL_MPS, DESIRED_PITCH_DEG

        state = _fresh_hold_state()
        # Trusted crawl speed below target → mild nose-down, not free-dive.
        _rr, _pr, _yr, _t, dbg = compute_guidance(
            roll_deg=0.0,
            pitch_deg=0.0,
            quat=self._level_quat(),
            vY=0.0,
            vD=0.0,
            vision=None,
            vision_vel=None,
            state=state,
            vX=0.6,
        )
        self.assertFalse(dbg["vision_valid"])
        self.assertAlmostEqual(dbg["v_target"], BLIND_CRAWL_MPS, places=3)
        self.assertLessEqual(dbg["pitch_des_deg"], DESIRED_PITCH_DEG)
        self.assertGreaterEqual(dbg["pitch_des_deg"], -2.51)

    def test_weak_detection_reduces_bank(self):
        state = _fresh_hold_state()
        vision = {
            "frame_id": 12,
            "body_x_m": 12.0,
            "body_y_m": 0.8,
            "body_z_m": 0.0,
            "normal_body": None,
            "reliable": False,
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
            vX=1.0,
        )
        self.assertLess(abs(dbg["desired_roll"]), 8.0)
        self.assertLess(dbg["blend"], 0.5)

    def test_lean_ramp_blocks_min_dive_just_after_go(self):
        from simulator.gp_pilot import DESIRED_PITCH_DEG, PITCH_DES_MIN_DEG

        state = _fresh_hold_state()
        vision = {
            "frame_id": 1,
            "body_x_m": 12.0,
            "body_y_m": 0.0,
            "body_z_m": 0.0,
            "normal_body": None,
            "reliable": True,
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
            vX=0.0,
            flying_t=0.2,
        )
        self.assertGreaterEqual(dbg["pitch_des_deg"], DESIRED_PITCH_DEG - 0.01)
        self.assertGreaterEqual(dbg["pitch_des_deg"], PITCH_DES_MIN_DEG)

    def test_untrusted_vx_blocks_dive_immediately(self):
        """vX≈0 must not dive even past lean ramp (no delay before clamp)."""
        from simulator.gp_pilot import DESIRED_PITCH_DEG, PITCH_DES_MIN_DEG

        state = _fresh_hold_state()
        vision = {
            "frame_id": 1,
            "body_x_m": 12.0,
            "body_y_m": 0.0,
            "body_z_m": 0.0,
            "normal_body": None,
            "reliable": True,
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
            vX=0.0,
            flying_t=5.0,  # past lean ramp
        )
        self.assertAlmostEqual(dbg["pitch_des_deg"], DESIRED_PITCH_DEG, places=2)
        self.assertGreaterEqual(dbg["pitch_des_deg"], PITCH_DES_MIN_DEG)


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
                            "reproj_px": 1.5,
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
        self.assertEqual(est["source"], "yolo")

    def test_anduril_preferred_over_yolo(self):
        data = {
            "anduril_gate": {
                "frame_id": 3,
                "body_x_m": 7.0,
                "body_y_m": 0.1,
                "body_z_m": 0.0,
                "pnp_ok": True,
                "reliable": True,
                "source": "anduril",
                "normal_body": None,
            },
            "pose": {
                "frame_id": 3,
                "gates": [
                    {
                        "conf": 0.99,
                        "pose": {
                            "gate_pos_body": np.array([99.0, 0.0, 0.0]),
                            "normal_body": np.array([-1.0, 0.0, 0.0]),
                            "reproj_px": 1.0,
                        },
                    }
                ],
            },
        }
        est = vision_gate_estimate(data)
        self.assertEqual(est["source"], "anduril")
        self.assertAlmostEqual(est["body_x_m"], 7.0)

    def test_detect_gate_finds_red_square(self):
        import cv2

        from simulator.anduril_gate_detect import detect_gate

        img = np.zeros((360, 640, 3), dtype=np.uint8)
        # Saturated red square near image centre (HSV red near hue 0).
        cv2.rectangle(img, (280, 120), (360, 240), (0, 0, 255), -1)
        est, _mask, _cnts = detect_gate(img)
        self.assertIsNotNone(est)
        self.assertTrue(est["reliable"])
        self.assertGreater(est["area"], 300)

    def test_gate_smoother_emas_and_sticks_pnp(self):
        from simulator.gp_vision import GateEstimateSmoother

        sm = GateEstimateSmoother()
        data = {
            "pose": {
                "frame_id": 1,
                "gates": [
                    {
                        "conf": 0.95,
                        "pose": {
                            "gate_pos_body": np.array([10.0, 0.0, 0.0]),
                            "normal_body": np.array([-1.0, 0.0, 0.0]),
                            "reproj_px": 1.0,
                        },
                    }
                ],
            }
        }
        e1 = sm.update(data)
        self.assertAlmostEqual(e1["body_x_m"], 10.0, places=3)
        data["pose"]["frame_id"] = 2
        data["pose"]["gates"][0]["pose"]["gate_pos_body"] = np.array([8.0, 0.0, 0.0])
        e2 = sm.update(data)
        # EMA blends toward the new measurement, not a hard jump.
        self.assertLess(e2["body_x_m"], 10.0)
        self.assertGreater(e2["body_x_m"], 8.0)
        # Drop PnP; HSV-only must not replace a fresh PnP fix for a few frames.
        data = {
            "pose": {"frame_id": 3, "gates": []},
            "gate_target": {
                "detected": True,
                "frame_id": 3,
                "u_px": 320.0,
                "v_px": 180.0,
                "r_frac": 0.05,
            },
        }
        e3 = sm.update(data)
        self.assertTrue(e3["pnp_ok"])
        self.assertAlmostEqual(e3["body_x_m"], e2["body_x_m"], places=5)


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
        # Idle holds ride the same attitude-quat wire: level + zero thrust.
        args = ctrl.set_attitude_quat_deg.call_args[0]
        self.assertEqual(args, (0.0, 0.0, 0.0, 0.0))
        pilot.shutdown()

    def test_gp_selects_quat_wire_at_60hz(self):
        from simulator.gp_pilot import GP_CONTROL_HZ, GPPilot

        ctrl = MagicMock()
        pilot = GPPilot(ctrl, {"armed": False})
        # Original AndurilGP wire: attitude-quat encoding at 60 Hz.
        ctrl.set_control_mode.assert_called_with("attitude_quat")
        self.assertEqual(ctrl.control_hz, GP_CONTROL_HZ)
        self.assertEqual(GP_CONTROL_HZ, 60)
        pilot.shutdown()

    def test_attitude_quat_wire_format(self):
        from simulator.controller import ATT_QUAT_TYPE_MASK, _send_attitude_quat

        conn = MagicMock()
        conn.target_system = 1
        conn.target_component = 1
        _send_attitude_quat(
            conn, 0, roll_deg=10.0, pitch_deg=-5.0, yaw_deg=3.0, thrust=0.3
        )
        args = conn.mav.set_attitude_target_send.call_args[0]
        # (time, sys, comp, mask, q, roll_rate, pitch_rate, yaw_rate, thrust)
        self.assertEqual(args[3], 0b00000111)  # ignore rates, USE attitude
        self.assertEqual(ATT_QUAT_TYPE_MASK, 0b00000111)
        expected_q = euler_to_quat(
            math.radians(10.0), math.radians(-5.0), math.radians(3.0)
        )
        np.testing.assert_allclose(args[4], expected_q, atol=1e-12)
        self.assertEqual(args[5:8], (0.0, 0.0, 0.0))  # body rates zeroed
        self.assertEqual(args[8], 0.3)


class GpRaceGateTests(unittest.TestCase):
    """WAIT_FOR_START must fly fresh countdowns AND recover stale/reset races."""

    def _pilot(self):
        from simulator.gp_pilot import GPPilot

        ctrl = MagicMock()
        data = {"armed": True, "imu": {"time_us": 1}}
        pilot = GPPilot(ctrl, data)
        pilot._open_log = lambda: None  # keep unit tests out of rl/data
        return ctrl, data, pilot

    def _go_flying(self, ctrl, data, pilot):
        """Drive WAIT_FOR_START through a fresh countdown into FLYING."""
        from simulator.gp_pilot import Phase

        pilot.tick()  # armed + IMU -> WAIT_FOR_START
        data["race_status"] = {
            "sim_boot_time_ms": 1000,
            "race_start_boot_time_ms": -1,
            "race_finish_time_ns": -1,
        }
        pilot.tick()  # anchor
        data["race_status"] = {
            "sim_boot_time_ms": 5000,
            "race_start_boot_time_ms": 4000,
            "race_finish_time_ns": -1,
        }
        pilot.tick()
        self.assertEqual(pilot.phase, Phase.FLYING)
        return ctrl, data, pilot

    def test_flying_sends_degree_commands_on_quat_wire(self):
        from simulator.gp_pilot import Phase

        ctrl, data, pilot = self._pilot()
        try:
            self._go_flying(ctrl, data, pilot)
            pilot.tick()  # one FLYING tick, no vision: crawl speed loop
            roll_cmd, pitch_cmd, yaw_cmd, thrust = ctrl.set_attitude_quat_deg.call_args[
                0
            ]
            self.assertGreater(abs(pitch_cmd), 5.0)
            self.assertAlmostEqual(roll_cmd, 0.0, delta=0.5)
            self.assertGreater(thrust, 0.2)
            self.assertLess(thrust, 0.4)
        finally:
            pilot.shutdown()

    def test_fresh_race_go_flies(self):
        from simulator.gp_pilot import Phase

        ctrl, data, pilot = self._pilot()
        try:
            pilot.tick()  # armed + IMU -> WAIT_FOR_START
            data["race_status"] = {
                "sim_boot_time_ms": 1000,
                "race_start_boot_time_ms": -1,
            }
            pilot.tick()  # anchor at 1000
            data["race_status"] = {
                "sim_boot_time_ms": 5000,
                "race_start_boot_time_ms": 4000,
            }
            pilot.tick()
            self.assertEqual(pilot.phase, Phase.FLYING)
            ctrl.send_sim_reset_command.assert_not_called()
        finally:
            pilot.shutdown()

    def test_stale_race_does_not_fly_without_fresh_countdown(self):
        """After anchor, a race_start *before* the anchor must NOT early-fly.

        Manual Restart Race needs the full 3s countdown (fresh start_ms).
        """
        from simulator.gp_pilot import Phase

        ctrl, data, pilot = self._pilot()
        try:
            pilot.tick()
            # Client anchors late; old race_start is stale vs anchor.
            data["race_status"] = {
                "sim_boot_time_ms": 433289,
                "race_start_boot_time_ms": 3307,
                "race_finish_time_ns": -1,
            }
            pilot.tick()
            self.assertEqual(pilot.phase, Phase.WAIT_FOR_START)
            # Mid-countdown of a *fresh* restart: start in the future.
            data["race_status"] = {
                "sim_boot_time_ms": 433289,
                "race_start_boot_time_ms": 436289,  # 3s ahead
                "race_finish_time_ns": -1,
            }
            pilot.tick()
            self.assertEqual(pilot.phase, Phase.WAIT_FOR_START)
            # Countdown elapsed.
            data["race_status"] = {
                "sim_boot_time_ms": 436300,
                "race_start_boot_time_ms": 436289,
                "race_finish_time_ns": -1,
            }
            pilot.tick()
            self.assertEqual(pilot.phase, Phase.FLYING)
        finally:
            pilot.shutdown()

    def test_mid_countdown_stays_in_wait(self):
        from simulator.gp_pilot import Phase

        ctrl, data, pilot = self._pilot()
        try:
            pilot.tick()
            data["race_status"] = {
                "sim_boot_time_ms": 1000,
                "race_start_boot_time_ms": -1,
            }
            pilot.tick()
            data["race_status"] = {
                "sim_boot_time_ms": 2000,
                "race_start_boot_time_ms": 4000,  # GO still 2s away
                "race_finish_time_ns": -1,
            }
            for _ in range(3):
                pilot.tick()
            self.assertEqual(pilot.phase, Phase.WAIT_FOR_START)
            args = ctrl.set_attitude_quat_deg.call_args[0]
            self.assertEqual(args, (0.0, 0.0, 0.0, 0.0))
        finally:
            pilot.shutdown()

    def test_restart_race_while_flying_returns_to_wait(self):
        from simulator.gp_pilot import Phase

        ctrl, data, pilot = self._pilot()
        try:
            pilot.tick()
            data["race_status"] = {
                "sim_boot_time_ms": 1000,
                "race_start_boot_time_ms": -1,
            }
            pilot.tick()
            data["race_status"] = {
                "sim_boot_time_ms": 5000,
                "race_start_boot_time_ms": 4000,
                "race_finish_time_ns": -1,
            }
            pilot.tick()
            self.assertEqual(pilot.phase, Phase.FLYING)
            # Restart Race: new countdown (start in the future).
            data["race_status"] = {
                "sim_boot_time_ms": 8000,
                "race_start_boot_time_ms": 11000,
                "race_finish_time_ns": -1,
            }
            pilot.tick()
            self.assertEqual(pilot.phase, Phase.WAIT_FOR_START)
            args = ctrl.set_attitude_quat_deg.call_args[0]
            self.assertEqual(args, (0.0, 0.0, 0.0, 0.0))
        finally:
            pilot.shutdown()

    def test_sim_clock_reset_reanchors_then_flies_new_countdown(self):
        from simulator.gp_pilot import Phase

        ctrl, data, pilot = self._pilot()
        try:
            pilot.tick()
            data["race_status"] = {
                "sim_boot_time_ms": 433289,
                "race_start_boot_time_ms": -1,
            }
            pilot.tick()  # anchored high, no race yet
            # User clicks Restart Race: sim clock rewinds. Must re-arm +
            # re-anchor, not stay gated off by the stale 433289 anchor.
            data["race_status"] = {
                "sim_boot_time_ms": 400,
                "race_start_boot_time_ms": -1,
            }
            pilot.tick()
            self.assertEqual(pilot.phase, Phase.WAIT_FOR_DATA)
            pilot.tick()  # armed + IMU -> back to WAIT_FOR_START
            data["race_status"] = {
                "sim_boot_time_ms": 900,
                "race_start_boot_time_ms": -1,
            }
            pilot.tick()  # fresh anchor at 900
            data["race_status"] = {
                "sim_boot_time_ms": 4000,
                "race_start_boot_time_ms": 3500,
            }
            pilot.tick()
            self.assertEqual(pilot.phase, Phase.FLYING)
        finally:
            pilot.shutdown()

    def test_frozen_imu_clock_notes_idle_physics_and_recovers(self):
        import simulator.gp_pilot as gp

        ctrl, data, pilot = self._pilot()
        try:
            self._go_flying(ctrl, data, pilot)
            with patch.object(gp, "IMU_FROZEN_S", -1.0):
                pilot.tick()  # same ts, past threshold -> idle note
            self.assertTrue(pilot._frozen_noted)
            data["imu"] = {"time_us": 2}  # clock moves: physics live again
            pilot.tick()
            self.assertFalse(pilot._frozen_noted)
        finally:
            pilot.shutdown()

    def test_frozen_physics_blocks_flying_entry(self):
        import simulator.gp_pilot as gp
        from simulator.gp_pilot import Phase

        ctrl, data, pilot = self._pilot()
        try:
            with patch.object(gp, "IMU_FROZEN_S", -1.0):
                pilot.tick()  # -> WAIT_FOR_START
                data["race_status"] = {
                    "sim_boot_time_ms": 1000,
                    "race_start_boot_time_ms": -1,
                    "race_finish_time_ns": -1,
                }
                pilot.tick()
                data["race_status"] = {
                    "sim_boot_time_ms": 5000,
                    "race_start_boot_time_ms": 4000,
                    "race_finish_time_ns": -1,
                }
                pilot.tick()
                pilot.tick()
                self.assertEqual(pilot.phase, Phase.WAIT_FOR_START)
            data["imu"] = {"time_us": 2}  # release: clock advances
            pilot.tick()
            self.assertEqual(pilot.phase, Phase.FLYING)
        finally:
            pilot.shutdown()

    def test_disarm_blip_mid_flight_keeps_flying(self):
        from simulator.gp_pilot import Phase

        ctrl, data, pilot = self._pilot()
        try:
            self._go_flying(ctrl, data, pilot)
            data["armed"] = False
            pilot.tick()
            self.assertEqual(pilot.phase, Phase.FLYING)
            data["armed"] = True
            pilot.tick()
            self.assertEqual(pilot.phase, Phase.FLYING)
            self.assertIsNone(pilot._disarm_since)
        finally:
            pilot.shutdown()

    def test_persistent_disarm_mid_flight_recycles_to_arm_phase(self):
        import simulator.gp_pilot as gp
        from simulator.gp_pilot import Phase

        ctrl, data, pilot = self._pilot()
        try:
            self._go_flying(ctrl, data, pilot)
            data["armed"] = False
            with patch.object(gp, "DISARM_PERSIST_S", 0.0):
                pilot.tick()
                pilot.tick()
            self.assertEqual(pilot.phase, Phase.WAIT_FOR_DATA)
        finally:
            pilot.shutdown()

    def test_finished_race_holds_and_instructs_restart(self):
        from simulator.gp_pilot import Phase

        ctrl, data, pilot = self._pilot()
        try:
            pilot.tick()
            data["race_status"] = {
                "sim_boot_time_ms": 433289,
                "race_start_boot_time_ms": 3307,
                "race_finish_time_ns": 120_000_000_000,
            }
            pilot.tick()
            pilot.tick()
            self.assertEqual(pilot.phase, Phase.WAIT_FOR_START)
            ctrl.send_sim_reset_command.assert_not_called()
        finally:
            pilot.shutdown()

    def test_disarm_during_wait_returns_to_arm_phase(self):
        from simulator.gp_pilot import Phase

        ctrl, data, pilot = self._pilot()
        try:
            pilot.tick()
            self.assertEqual(pilot.phase, Phase.WAIT_FOR_START)
            data["armed"] = False  # sim reset disarms the drone
            pilot.tick()
            self.assertEqual(pilot.phase, Phase.WAIT_FOR_DATA)
        finally:
            pilot.shutdown()

    def test_collision_enters_backoff_then_resumes(self):
        import simulator.gp_pilot as gp
        from simulator.gp_pilot import Phase

        ctrl, data, pilot = self._pilot()
        try:
            self._go_flying(ctrl, data, pilot)
            data["collision"] = {"id": 1, "threat_level": 1, "delta": 0.0}
            with patch.object(gp, "BACKOFF_DIST_M", 100.0), patch.object(
                gp, "BACKOFF_MAX_S", 100.0
            ):
                pilot.tick()
                self.assertEqual(pilot.phase, Phase.BACKOFF)
                self.assertIsNone(data.get("collision"))
                _r, pitch_cmd, _y, thrust = ctrl.set_attitude_quat_deg.call_args[0]
                self.assertGreater(pitch_cmd, 0.0)
                self.assertGreater(thrust, 0.2)
            with patch.object(gp, "BACKOFF_MIN_S", 0.0), patch.object(
                gp, "BACKOFF_DIST_M", 0.0
            ):
                pilot.tick()
            self.assertEqual(pilot.phase, Phase.FLYING)
        finally:
            pilot.shutdown()

    def test_backoff_levels_pitch_when_reverse_overspeed(self):
        """Reverse > BACKOFF_MAX_SPEED must not keep commanding hard nose-up."""
        from simulator.gp_pilot import BACKOFF_PITCH_DEG, Phase

        ctrl, data, pilot = self._pilot()
        try:
            self._go_flying(ctrl, data, pilot)
            pilot._enter_backoff()
            ctrl.set_attitude_quat_deg.reset_mock()
            pilot._tick_backoff(
                roll_deg=0.0,
                pitch_deg=0.0,
                yaw_deg=0.0,
                vX=-3.0,  # ~11 km/h reverse
                vY=0.0,
                vD=0.0,
                dt=1.0 / 60.0,
            )
            _r, pitch_cmd, _y, _t = ctrl.set_attitude_quat_deg.call_args[0]
            self.assertLess(abs(pitch_cmd), 0.5)
            self.assertLess(pitch_cmd, 0.5 * BACKOFF_PITCH_DEG)
        finally:
            pilot.shutdown()

    def test_backoff_nose_up_when_reverse_slow(self):
        from simulator.gp_pilot import Phase

        ctrl, data, pilot = self._pilot()
        try:
            self._go_flying(ctrl, data, pilot)
            pilot._enter_backoff()
            ctrl.set_attitude_quat_deg.reset_mock()
            pilot._tick_backoff(
                roll_deg=0.0,
                pitch_deg=0.0,
                yaw_deg=0.0,
                vX=0.0,
                vY=0.0,
                vD=0.0,
                dt=1.0 / 60.0,
            )
            _r, pitch_cmd, _y, _t = ctrl.set_attitude_quat_deg.call_args[0]
            self.assertGreater(pitch_cmd, 0.0)
            self.assertEqual(pilot.phase, Phase.BACKOFF)
        finally:
            pilot.shutdown()

    def test_backoff_resume_rearms_lean_ramp_and_resets_est(self):
        """Collision resume must match GO hygiene so vX≈0 cannot open-loop dive."""
        import simulator.gp_pilot as gp
        from simulator.gp_pilot import DESIRED_PITCH_DEG, PITCH_DES_MIN_DEG, Phase

        ctrl, data, pilot = self._pilot()
        try:
            self._go_flying(ctrl, data, pilot)
            flying_since_go = pilot._flying_since
            # Age past lean ramp so a bare FLYING resume would skip it.
            pilot._flying_since = time.time() - 5.0
            pilot.est.vel_body[:] = 2.5
            pilot.est.vel_ned[:] = 2.5

            data["collision"] = {"id": 1, "threat_level": 1, "delta": 0.0}
            with patch.object(gp, "BACKOFF_DIST_M", 100.0), patch.object(
                gp, "BACKOFF_MAX_S", 100.0
            ):
                pilot.tick()
                self.assertEqual(pilot.phase, Phase.BACKOFF)

            with (
                patch.object(gp, "BACKOFF_MIN_S", 0.0),
                patch.object(gp, "BACKOFF_DIST_M", 0.0),
                patch.object(pilot.est, "reset", wraps=pilot.est.reset) as rst,
            ):
                pilot.tick()

            self.assertEqual(pilot.phase, Phase.FLYING)
            rst.assert_called()
            self.assertIsNotNone(pilot._flying_since)
            self.assertGreater(pilot._flying_since, flying_since_go)
            self.assertLess(time.time() - pilot._flying_since, 0.5)
            snap = pilot.est.snapshot()
            self.assertAlmostEqual(float(snap["vel_body"][0]), 0.0, places=5)

            # Guidance with vX=0 just after resume must not saturate min dive.
            from simulator.gp_pilot import _fresh_hold_state, compute_guidance

            _rr, _pr, _yr, _t, dbg = compute_guidance(
                roll_deg=0.0,
                pitch_deg=0.0,
                quat=np.array([1.0, 0.0, 0.0, 0.0]),
                vY=0.0,
                vD=0.0,
                vision={
                    "frame_id": 1,
                    "body_x_m": 12.0,
                    "body_y_m": 0.0,
                    "body_z_m": 0.0,
                    "normal_body": None,
                    "reliable": True,
                },
                vision_vel=None,
                state=_fresh_hold_state(),
                vX=0.0,
                flying_t=time.time() - pilot._flying_since,
            )
            self.assertGreaterEqual(dbg["pitch_des_deg"], DESIRED_PITCH_DEG - 0.01)
            self.assertGreater(dbg["pitch_des_deg"], PITCH_DES_MIN_DEG)
        finally:
            pilot.shutdown()

    def test_abort_to_wait_resets_estimator(self):
        from simulator.gp_pilot import Phase

        ctrl, data, pilot = self._pilot()
        try:
            self._go_flying(ctrl, data, pilot)
            pilot.est.vel_body[:] = 3.0
            with patch.object(pilot.est, "reset", wraps=pilot.est.reset) as rst:
                data["race_status"] = {
                    "sim_boot_time_ms": 5000,
                    "race_start_boot_time_ms": 6000,  # future = new countdown
                    "race_finish_time_ns": -1,
                }
                pilot.tick()
            self.assertEqual(pilot.phase, Phase.WAIT_FOR_START)
            rst.assert_called()
            self.assertAlmostEqual(float(pilot.est.snapshot()["vel_body"][0]), 0.0)
        finally:
            pilot.shutdown()


class EstimatorResilienceTests(unittest.TestCase):
    def test_estimator_thread_survives_malformed_imu(self):
        from simulator.gp_estimation import GPEstimation

        def imu(t_us, gz=0.0):
            return {
                "gx": 0.0,
                "gy": 0.0,
                "gz": gz,
                "ax": 0.0,
                "ay": 0.0,
                "az": -9.81,
                "time_us": t_us,
            }

        data = {"imu": {"time_us": "garbage"}}  # int() raises ValueError
        est = GPEstimation(data, launch_pitch_deg=0.0)
        with patch("simulator.gp_estimation.traceback.print_exc") as pe:
            est.start()
            try:
                time.sleep(0.1)  # loop must chew on the bad sample and live
                self.assertTrue(est._thread.is_alive())
                self.assertTrue(pe.called)

                # Valid samples must then integrate normally: 0.5 rad/s yaw
                # for 0.1 s (sign-negated) -> ~ -2.9 deg.
                data["imu"] = imu(1_000_000)
                time.sleep(0.05)
                data["imu"] = imu(1_100_000, gz=0.5)
                deadline = time.monotonic() + 2.0
                yaw = 0.0
                while time.monotonic() < deadline:
                    yaw = est.snapshot()["att_deg"][2]
                    if abs(yaw) > 0.5:
                        break
                    time.sleep(0.01)
                self.assertTrue(est._thread.is_alive())
                self.assertGreater(abs(yaw), 0.5)
            finally:
                est.stop()


if __name__ == "__main__":
    unittest.main()
