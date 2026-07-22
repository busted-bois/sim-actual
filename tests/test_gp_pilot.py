"""Unit tests for AndurilGP controls port (AHRS, guidance, wiring)."""

import math
import time
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from simulator.gp_pilot import (
    ELEV_I_SEED,
    HOVER_THRUST,
    K_BEARING,
    MAX_BANK_DEG,
    MAX_DESCENT_RATE_MPS,
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

    def _descend_thrust(self, vD, floor_clearance=float("nan")):
        state = _fresh_hold_state()
        vision = {
            "frame_id": 5,
            "body_x_m": 10.0,
            "body_y_m": 0.0,
            "body_z_m": 2.0,  # gate below → thrust cut to descend
            "normal_body": None,
        }
        _rr, _pr, _yr, thrust, _dbg = compute_guidance(
            roll_deg=0.0,
            pitch_deg=0.0,
            quat=self._level_quat(),
            vY=0.0,
            vD=vD,
            vision=vision,
            vision_vel=None,
            state=state,
            vX=2.0,
            floor_clearance_m=floor_clearance,
        )
        return thrust

    def test_descent_rate_capped(self):
        # Sinking well past the cap: the thrust cut for a below-gate must be
        # clamped back up to the hover trim so the sink can't keep building
        # into the bottom bar.
        slow = self._descend_thrust(vD=0.0)  # not sinking: full cut allowed
        fast = self._descend_thrust(vD=MAX_DESCENT_RATE_MPS + 1.0)  # over cap
        self.assertLess(slow, HOVER_THRUST)  # descend command present
        self.assertGreater(fast, slow)  # cap raised thrust back up
        self.assertAlmostEqual(fast, HOVER_THRUST + ELEV_I_SEED, places=6)

    def test_floor_guard_climbs_near_ground(self):
        # Blind (no gate), near the floor: climb thrust must blend in.
        state = _fresh_hold_state()

        def blind_thrust(clearance):
            _rr, _pr, _yr, thrust, _dbg = compute_guidance(
                roll_deg=0.0,
                pitch_deg=0.0,
                quat=self._level_quat(),
                vY=0.0,
                vD=0.0,
                vision=None,
                vision_vel=None,
                state=dict(state),
                vX=2.0,
                floor_clearance_m=clearance,
            )
            return thrust

        safe = blind_thrust(5.0)  # well above floor: guard inert
        low = blind_thrust(0.2)  # near floor: guard climbs
        self.assertGreater(low, safe)
        self.assertGreater(low, HOVER_THRUST + ELEV_I_SEED)

    def test_floor_guard_suppressed_when_gate_below(self):
        # A fresh gate genuinely below us (intended descent) must NOT trip the
        # floor guard — otherwise the descending course can't be flown.
        state = _fresh_hold_state()
        vision = {
            "frame_id": 5,
            "body_x_m": 10.0,
            "body_y_m": 0.0,
            "body_z_m": 2.0,  # gate 2 m below → descend toward it
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
            vX=2.0,
            floor_clearance_m=0.2,  # low, but a gate is below → not a fall
        )
        self.assertGreater(dbg["elev_err"], 0.3)
        self.assertLess(thrust, HOVER_THRUST)  # descend command preserved

    def test_elev_integral_trims_persistent_low_offset(self):
        # Drone stuck 0.5 m below gate centre (hover-trim mismatch): the I-term
        # must keep raising thrust over time where P-only plateaued.
        state = _fresh_hold_state()

        def tick(fid):
            vision = {
                "frame_id": fid,
                "body_x_m": 8.0,
                "body_y_m": 0.0,
                "body_z_m": -0.5,  # gate above → climb
                "normal_body": None,
            }
            _rr, _pr, _yr, thrust, _dbg = compute_guidance(
                roll_deg=0.0,
                pitch_deg=0.0,
                quat=self._level_quat(),
                vY=0.0,
                vD=0.0,
                vision=vision,
                vision_vel=None,
                state=state,
            )
            return thrust

        t_first = tick(1)
        for fid in range(2, 121):  # ~2 s at 60 Hz
            t_last = tick(fid)
        self.assertGreater(state["elev_i"], 0.003)
        self.assertGreater(t_last, t_first + 0.002)
        # Clamp: integral can never run away.
        for fid in range(121, 2000):
            tick(fid)
        from simulator.gp_pilot import ELEV_I_CLAMP

        self.assertLessEqual(state["elev_i"], ELEV_I_CLAMP + 1e-9)

    def test_elev_i_does_not_wind_up_on_large_error(self):
        # Gate-1 climb-out: metres of elev error for seconds must NOT charge
        # the integrator (it hit the clamp and ballooned over the gate).
        state = _fresh_hold_state()
        vision = {
            "frame_id": 1,
            "body_x_m": 8.0,
            "body_y_m": 0.0,
            "body_z_m": -2.5,  # far above — transient, not trim
            "normal_body": None,
        }
        for fid in range(1, 121):
            vision = dict(vision, frame_id=fid)
            compute_guidance(
                roll_deg=0.0,
                pitch_deg=0.0,
                quat=self._level_quat(),
                vY=0.0,
                vD=0.0,
                vision=vision,
                vision_vel=None,
                state=state,
            )
        from simulator.gp_pilot import ELEV_I_SEED

        self.assertAlmostEqual(state["elev_i"], ELEV_I_SEED, places=9)

    def test_track_break_clears_derivative_state(self):
        # A gate switch must not inject phantom bearing/elev rates.
        state = _fresh_hold_state()
        v1 = {
            "frame_id": 1,
            "body_x_m": 6.0,
            "body_y_m": -0.5,
            "body_z_m": 0.0,
            "normal_body": None,
        }
        compute_guidance(
            roll_deg=0.0,
            pitch_deg=0.0,
            quat=self._level_quat(),
            vY=0.0,
            vD=0.0,
            vision=v1,
            vision_vel=None,
            state=state,
        )
        v2 = dict(v1, frame_id=2, body_y_m=1.5, track_break=True)
        _rr, _pr, _yr, _t, dbg = compute_guidance(
            roll_deg=0.0,
            pitch_deg=0.0,
            quat=self._level_quat(),
            vY=0.0,
            vD=0.0,
            vision=v2,
            vision_vel=None,
            state=state,
        )
        # Without the clear, the 2 m by-jump reads ~570 deg/s bearing rate
        # and pins a phantom d_lat; with it, the D-term starts fresh.
        self.assertAlmostEqual(dbg["d_lat"], 0.0, places=6)

    def test_blind_sideslip_null_banks_against_drift(self):
        state = _fresh_hold_state()
        _rr, _pr, _yr, _t, dbg = compute_guidance(
            roll_deg=0.0,
            pitch_deg=0.0,
            quat=self._level_quat(),
            vY=0.5,  # drifting right while blind
            vD=0.0,
            vision=None,
            vision_vel=None,
            state=state,
        )
        self.assertAlmostEqual(dbg["desired_roll"], -4.0, places=3)  # bank left

    def test_blind_elev_err_decays(self):
        state = _fresh_hold_state()
        state["last_elev_err"] = 1.0
        compute_guidance(
            roll_deg=0.0,
            pitch_deg=0.0,
            quat=self._level_quat(),
            vY=0.0,
            vD=0.0,
            vision=None,
            vision_vel=None,
            state=state,
        )
        self.assertAlmostEqual(state["last_elev_err"], 0.97, places=4)

    def test_blind_sink_null_raises_thrust(self):
        # Vertical analog of the sideslip null: residual sink while blind must
        # be actively arrested, not integrated open-loop into the bottom bar
        # (log: 0.7 m/s sink at lost-vision cost 0.47 m over a 0.65 s window).
        from simulator.gp_pilot import ELEV_I_SEED

        state = _fresh_hold_state()
        _rr, _pr, _yr, thrust, dbg = compute_guidance(
            roll_deg=0.0,
            pitch_deg=0.0,
            quat=self._level_quat(),
            vY=0.0,
            vD=0.7,  # sinking while blind
            vision=None,
            vision_vel=None,
            state=state,
        )
        self.assertAlmostEqual(dbg["d_vert"], 0.7, places=6)
        self.assertGreater(thrust, HOVER_THRUST + ELEV_I_SEED + 0.02)

    def test_blind_climb_also_nulled(self):
        from simulator.gp_pilot import ELEV_I_SEED

        state = _fresh_hold_state()
        _rr, _pr, _yr, thrust, dbg = compute_guidance(
            roll_deg=0.0,
            pitch_deg=0.0,
            quat=self._level_quat(),
            vY=0.0,
            vD=-0.7,  # ballooning up while blind
            vision=None,
            vision_vel=None,
            state=state,
        )
        self.assertAlmostEqual(dbg["d_vert"], -0.7, places=6)
        self.assertLess(thrust, HOVER_THRUST + ELEV_I_SEED - 0.02)

    def test_blind_vd_null_clamped(self):
        from simulator.gp_pilot import BLIND_VD_CLAMP_MPS

        state = _fresh_hold_state()
        _rr, _pr, _yr, _t, dbg = compute_guidance(
            roll_deg=0.0,
            pitch_deg=0.0,
            quat=self._level_quat(),
            vY=0.0,
            vD=4.0,  # IMU dead-reckoning can drift — bound the damage
            vision=None,
            vision_vel=None,
            state=state,
        )
        self.assertAlmostEqual(dbg["d_vert"], BLIND_VD_CLAMP_MPS, places=6)

    def test_near_gate_valid_vision_still_nulls_sink(self):
        # Inside MIN_BX_FOR_ELEV with vision VALID the old code zeroed the
        # vertical D entirely (elev_rate gated off) — the null must cover
        # this window too, not just full blindness.
        state = _fresh_hold_state()
        vision = {
            "frame_id": 3,
            "body_x_m": 2.2,  # < MIN_BX_FOR_ELEV
            "body_y_m": 0.0,
            "body_z_m": 0.0,
            "normal_body": None,
        }
        _rr, _pr, _yr, _t, dbg = compute_guidance(
            roll_deg=0.0,
            pitch_deg=0.0,
            quat=self._level_quat(),
            vY=0.0,
            vD=0.6,
            vision=vision,
            vision_vel=None,
            state=state,
        )
        self.assertTrue(dbg["vision_valid"])
        self.assertAlmostEqual(dbg["d_vert"], 0.6, places=6)

    def test_near_gate_valid_vision_decays_frozen_elev_err(self):
        # The valid-but-close window used to HOLD the frozen error un-decayed.
        state = _fresh_hold_state()
        state["last_elev_err"] = 1.0
        vision = {
            "frame_id": 3,
            "body_x_m": 2.2,
            "body_y_m": 0.0,
            "body_z_m": 0.0,
            "normal_body": None,
        }
        compute_guidance(
            roll_deg=0.0,
            pitch_deg=0.0,
            quat=self._level_quat(),
            vY=0.0,
            vD=0.0,
            vision=vision,
            vision_vel=None,
            state=state,
        )
        self.assertAlmostEqual(state["last_elev_err"], 0.97, places=4)

    def test_vertical_error_slows_approach(self):
        # Unconverged vertical state must trade speed for settle time: same
        # range, big elev error => v_target drops to the THRU crawl.
        from simulator.gp_pilot import THRU_SPEED_MPS

        def run(bz):
            state = _fresh_hold_state()
            vision = {
                "frame_id": 3,
                "body_x_m": 4.0,
                "body_y_m": 0.0,
                "body_z_m": bz,
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
                vX=1.5,
                flying_t=10.0,
            )
            return dbg["v_target"]

        v_centred = run(0.0)
        v_low = run(2.0)  # 2 m below the gate line
        self.assertGreater(v_centred, v_low + 0.3)
        self.assertAlmostEqual(v_low, THRU_SPEED_MPS, places=6)

    def test_gate_tilt_head_on_reads_zero(self):
        from simulator.gp_vision import gate_tilt_deg_from_normal

        # Detector convention (normal points back at drone): head-on = (-1,0,0).
        # Regression: used to read ±180 → clip to a sign-flapping ±30 dither.
        self.assertAlmostEqual(
            gate_tilt_deg_from_normal(np.array([-1.0, 0.0, 0.0])), 0.0, places=6
        )
        # rl.gp_expert convention (normal points forward): also ~0 head-on.
        self.assertAlmostEqual(
            gate_tilt_deg_from_normal(np.array([1.0, 0.0, 0.0])), 0.0, places=6
        )
        # Fly-through axis 10° to the RIGHT → +10 (yaw right), both facings.
        th = math.radians(10.0)
        f = np.array([math.cos(th), math.sin(th), 0.0])
        self.assertAlmostEqual(gate_tilt_deg_from_normal(-f), 10.0, places=4)
        self.assertAlmostEqual(gate_tilt_deg_from_normal(f), 10.0, places=4)

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
        # Hover + the seeded trim (measured hover ~0.270 vs the 0.264 const).
        from simulator.gp_pilot import ELEV_I_SEED

        self.assertAlmostEqual(thrust, HOVER_THRUST + ELEV_I_SEED, places=2)

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

    @staticmethod
    def _anduril_plus_yolo(pose_fid=3, latest_fid=None):
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
                "frame_id": pose_fid,
                "gates": [
                    {
                        "conf": 0.99,
                        "pose": {
                            "gate_pos_body": np.array([9.0, 0.0, 0.0]),
                            "normal_body": np.array([-1.0, 0.0, 0.0]),
                            "reproj_px": 1.0,
                        },
                    }
                ],
            },
        }
        if latest_fid is not None:
            data["frame"] = {"frame_id": latest_fid}
        return data

    def test_fresh_yolo_preferred_over_anduril(self):
        data = self._anduril_plus_yolo(pose_fid=3, latest_fid=4)
        est = vision_gate_estimate(data)
        self.assertEqual(est["source"], "yolo")
        self.assertAlmostEqual(est["body_x_m"], 9.0)

    def test_stale_yolo_falls_back_to_anduril(self):
        data = self._anduril_plus_yolo(pose_fid=3, latest_fid=10)
        est = vision_gate_estimate(data)
        self.assertEqual(est["source"], "anduril")
        self.assertAlmostEqual(est["body_x_m"], 7.0)

    def test_best_pose_gate_picks_nearest_ahead(self):
        from simulator.gp_vision import best_pose_gate

        # Far gate dead ahead with HIGHER conf vs near gate slightly off-axis:
        # handoff must chase the near one.
        data = {
            "pose": {
                "frame_id": 1,
                "gates": [
                    {
                        "conf": 0.99,
                        "pose": {
                            "gate_pos_body": np.array([15.0, 0.0, 0.0]),
                            "normal_body": np.array([-1.0, 0.0, 0.0]),
                            "reproj_px": 1.0,
                        },
                    },
                    {
                        "conf": 0.6,
                        "pose": {
                            "gate_pos_body": np.array([6.0, 2.0, 0.0]),
                            "normal_body": np.array([-1.0, 0.0, 0.0]),
                            "reproj_px": 1.0,
                        },
                    },
                ],
            }
        }
        g = best_pose_gate(data)
        self.assertAlmostEqual(float(g["pose"]["gate_pos_body"][0]), 6.0)

    def test_best_pose_gate_ignores_third_gate(self):
        from simulator.gp_vision import best_pose_gate

        # Only the 2 NEAREST gates are eligible. A far gate dead-ahead (low
        # bearing) would otherwise win on cost over the two off-axis near gates
        # and yank the aim toward a gate two ahead while threading this one.
        def gate(bx, by, conf=0.9):
            return {
                "conf": conf,
                "pose": {
                    "gate_pos_body": np.array([bx, by, 0.0]),
                    "normal_body": np.array([-1.0, 0.0, 0.0]),
                    "reproj_px": 1.0,
                },
            }

        data = {
            "pose": {
                "frame_id": 1,
                "gates": [
                    gate(3.0, 2.6),  # nearest (r~3.97), off-axis -> cost ~7.5
                    gate(4.0, 3.0),  # 2nd     (r=5.0),  off-axis -> cost ~8.2
                    gate(6.0, 0.0),  # 3rd     (r=6.0),  dead-ahead cost 6.0
                ],
            }
        }
        g = best_pose_gate(data)
        # Without the 2-gate cap the dead-ahead far gate (cost 6.0) would win;
        # capped, only the two nearest are scored and the nearest one wins.
        self.assertAlmostEqual(float(g["pose"]["gate_pos_body"][0]), 3.0)

    def test_yolo_tracker_suppresses_near_gate_and_cools_down(self):
        from simulator.gp_vision import YOLO_PASS_COOLDOWN_FR, YoloGateTracker

        def pose_data(fid, bx):
            return {
                "frame_id": fid,
                "gates": [
                    {
                        "conf": 0.9,
                        "pose": {
                            "gate_pos_body": np.array([bx, 0.0, 0.0]),
                            "normal_body": np.array([-1.0, 0.0, 0.0]),
                            "reproj_px": 1.0,
                        },
                    }
                ],
            }

        tr = YoloGateTracker()
        # Approaching: normal estimate.
        est, sup = tr.update({"pose": pose_data(1, 8.0), "frame": {"frame_id": 1}})
        self.assertFalse(sup)
        self.assertEqual(est["source"], "yolo")
        # Threading the gate: blind, and suppression blocks ALL sources.
        est, sup = tr.update({"pose": pose_data(2, 1.0), "frame": {"frame_id": 2}})
        self.assertIsNone(est)
        self.assertTrue(sup)
        data = {"pose": pose_data(3, 1.2), "frame": {"frame_id": 3}}
        data["anduril_gate"] = {
            "frame_id": 3,
            "body_x_m": 1.2,
            "body_y_m": 0.0,
            "body_z_m": 0.0,
        }
        self.assertIsNone(vision_gate_estimate(data, tr))
        # Past the gate (next gate far ahead): cooldown holds for a while.
        est, sup = tr.update({"pose": pose_data(4, 12.0), "frame": {"frame_id": 4}})
        self.assertIsNone(est)
        self.assertTrue(sup)
        # After the cooldown expires, re-acquire the next gate.
        fid = 4 + YOLO_PASS_COOLDOWN_FR
        est, sup = tr.update({"pose": pose_data(fid, 12.0), "frame": {"frame_id": fid}})
        self.assertFalse(sup)
        self.assertAlmostEqual(est["body_x_m"], 12.0)

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


def _pose_data(fid, bx, by=0.0, bz=0.0):
    """YOLO pose packet + matching camera frame id."""
    return {
        "pose": {
            "frame_id": fid,
            "gates": [
                {
                    "conf": 0.9,
                    "pose": {
                        "gate_pos_body": np.array([bx, by, bz]),
                        "normal_body": np.array([-1.0, 0.0, 0.0]),
                        "reproj_px": 1.0,
                    },
                }
            ],
        },
        "frame": {"frame_id": fid},
    }


class SmootherIdentityTests(unittest.TestCase):
    def test_cooldown_survives_suppression(self):
        # Regression: smoother.reset() on every None estimate wiped the
        # tracker's in-gate state, so the post-pass cooldown never armed.
        from simulator.gp_vision import GateEstimateSmoother

        sm = GateEstimateSmoother()
        self.assertIsNotNone(sm.update(_pose_data(1, 8.0)))
        self.assertIsNone(sm.update(_pose_data(2, 1.0)))  # threading: blind
        # Past the gate: cooldown must hold even though est went None between.
        self.assertIsNone(sm.update(_pose_data(3, 12.0)))
        self.assertIsNone(sm.update(_pose_data(12, 12.0)))
        est = sm.update(_pose_data(13, 12.0))  # 3 + 10-frame cooldown elapsed
        self.assertIsNotNone(est)
        self.assertAlmostEqual(est["body_x_m"], 12.0)

    def test_identity_break_holds_then_snaps(self):
        from simulator.gp_vision import GateEstimateSmoother

        sm = GateEstimateSmoother()
        sm.update(_pose_data(1, 10.0))
        e2 = sm.update(_pose_data(2, 9.8))
        self.assertLess(e2["body_x_m"], 10.0)
        # A far challenger appears: hold the incumbent for 2 frames...
        e3 = sm.update(_pose_data(3, 25.0, by=5.0))
        self.assertLess(e3["body_x_m"], 11.0)  # still the incumbent
        self.assertNotIn("track_break", e3)
        e4 = sm.update(_pose_data(4, 25.0, by=5.0))
        self.assertLess(e4["body_x_m"], 11.0)
        # ...then SNAP to it (no EMA blend across identities) with the flag.
        e5 = sm.update(_pose_data(5, 25.0, by=5.0))
        self.assertAlmostEqual(e5["body_x_m"], 25.0)
        self.assertTrue(e5.get("track_break"))

    def test_near_far_handoff_goes_blind(self):
        # Incumbent tracked down to ~2.7 m, then only a far gate is reported:
        # that's a gate pass — blind cooldown, not an instant retarget.
        from simulator.gp_vision import GateEstimateSmoother

        sm = GateEstimateSmoother()
        fid = 0
        for bx in (8.0, 6.0, 4.5, 3.2, 2.5, 2.5, 2.5, 2.5, 2.5):
            fid += 1
            self.assertIsNotNone(sm.update(_pose_data(fid, bx)))
        for _ in range(3):  # challenger must persist BREAK_CONFIRM_N frames
            fid += 1
            out = sm.update(_pose_data(fid, 15.0))
        self.assertIsNone(out)  # handoff accepted → blind
        self.assertIsNone(sm.update(_pose_data(fid + 1, 15.0)))  # cooldown holds
        est = sm.update(_pose_data(fid + 10, 15.0))
        self.assertIsNotNone(est)
        self.assertAlmostEqual(est["body_x_m"], 15.0)

    def test_fid_regression_still_emas(self):
        # yolo carries the OLDER pose fid while anduril carries the camera
        # fid; raw fids regress on the flip, which used to bypass the EMA.
        from simulator.gp_vision import GateEstimateSmoother

        sm = GateEstimateSmoother()
        data1 = {
            "anduril_gate": {
                "frame_id": 10,
                "body_x_m": 8.0,
                "body_y_m": 0.0,
                "body_z_m": 0.0,
                "pnp_ok": False,
                "reliable": True,
                "source": "anduril",
                "normal_body": None,
            },
            "frame": {"frame_id": 10},
        }
        e1 = sm.update(data1)
        self.assertEqual(e1["frame_id"], 10)
        data2 = _pose_data(8, 7.0, by=0.3)
        data2["frame"] = {"frame_id": 11}  # pose fid 8 < anduril's 10
        e2 = sm.update(data2)
        self.assertEqual(e2["frame_id"], 11)  # monotonic camera clock
        # anduril -> yolo is a SOURCE FLIP: snap + track_break (blending
        # across estimators used to sweep the aim through their offset).
        self.assertAlmostEqual(e2["body_x_m"], 7.0, places=6)
        self.assertTrue(e2.get("track_break"))


class SourceFlipTests(unittest.TestCase):
    def test_source_flip_snaps_without_blending(self):
        # yolo -> anduril handoff: systematic offsets differ ~0.9 m vertically
        # on clipped views and the step passes _same_target — EMA-blending it
        # sweeps the aim point through the offset. Must SNAP + track_break.
        from simulator.gp_vision import GateEstimateSmoother

        sm = GateEstimateSmoother()

        def yolo_data(fid):
            return {
                "frame": {"frame_id": fid},
                "pose": {
                    "frame_id": fid,
                    "gates": [
                        {
                            "conf": 0.9,
                            "pose": {
                                "gate_pos_body": np.array([6.0, 0.0, 0.0]),
                                "normal_body": np.array([-1.0, 0.0, 0.0]),
                                "reproj_px": 1.0,
                            },
                        }
                    ],
                },
            }

        est = None
        for fid in range(1, 4):
            est = sm.update(yolo_data(fid))
        self.assertEqual(est["source"], "yolo")
        # YOLO stale (inference fell behind 6 frames); anduril takes over
        # 0.8 m higher within the EMA gap window (would blend without fix).
        data = {
            "frame": {"frame_id": 8},
            "pose": {"frame_id": 2, "gates": []},
            "anduril_gate": {
                "frame_id": 8,
                "body_x_m": 6.0,
                "body_y_m": 0.0,
                "body_z_m": -0.8,
                "source": "anduril",
                "reliable": True,
                "normal_body": None,
            },
        }
        est = sm.update(data)
        self.assertEqual(est["source"], "anduril")
        self.assertTrue(est.get("track_break"))
        self.assertAlmostEqual(est["body_z_m"], -0.8, places=6)  # snap, no EMA

    def test_source_flip_to_different_gate_debounces_not_snaps(self):
        # Regression: the source-flip snap must NOT bypass identity debounce.
        # A yolo->anduril flip that also lands on a DIFFERENT (far) gate used
        # to snap instantly — banking toward the far gate while threading the
        # near one. It must run the 3-frame BREAK_CONFIRM_N hold instead.
        from simulator.gp_vision import GateEstimateSmoother

        sm = GateEstimateSmoother()
        # Lock the incumbent on a near gate (bx~6) via YOLO for 3 frames.
        est = None
        for fid in range(1, 4):
            est = sm.update(_pose_data(fid, 6.0))
        self.assertEqual(est["source"], "yolo")

        def anduril_far(fid):
            return {
                "frame": {"frame_id": fid},
                "pose": {"frame_id": 2, "gates": []},  # yolo stale/absent
                "anduril_gate": {
                    "frame_id": fid,
                    "body_x_m": 15.0,
                    "body_y_m": 6.0,  # different range AND direction
                    "body_z_m": 0.0,
                    "source": "anduril",
                    "reliable": True,
                    "normal_body": None,
                },
            }

        # First flipped frame: HELD incumbent, not snapped to the far gate.
        e1 = sm.update(anduril_far(4))
        self.assertEqual(e1["source"], "yolo")
        self.assertAlmostEqual(e1["body_x_m"], 6.0, places=6)
        self.assertNotIn("track_break", e1)
        # Second frame still held.
        e2 = sm.update(anduril_far(5))
        self.assertAlmostEqual(e2["body_x_m"], 6.0, places=6)
        # Third confirming frame: NOW it may switch (debounce satisfied).
        e3 = sm.update(anduril_far(6))
        self.assertEqual(e3["source"], "anduril")
        self.assertAlmostEqual(e3["body_x_m"], 15.0, places=6)
        self.assertTrue(e3.get("track_break"))


class WiringTests(unittest.TestCase):
    def test_elev_i_persists_across_backoff_only(self):
        from simulator.gp_pilot import GPPilot

        pilot = GPPilot(MagicMock(), {})
        pilot._hold["elev_i"] = 0.012
        pilot._enter_backoff()
        self.assertAlmostEqual(pilot._hold["elev_i"], 0.012)
        pilot._resume_flying()
        self.assertAlmostEqual(pilot._hold["elev_i"], 0.012)
        pilot._reset_state()  # new race: trim restarts from the hover seed
        from simulator.gp_pilot import ELEV_I_SEED

        self.assertAlmostEqual(pilot._hold["elev_i"], ELEV_I_SEED)

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
        pilot._backoff_on = True  # backoff is opt-in now; test the mechanism
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

    def test_collision_default_no_backoff_keeps_flying(self):
        """Default (GP_BACKOFF off): a collision must NOT reverse — just consume
        the event and keep flying the guidance forward."""
        from simulator.gp_pilot import Phase

        ctrl, data, pilot = self._pilot()
        self.assertFalse(pilot._backoff_on)  # off by default
        try:
            self._go_flying(ctrl, data, pilot)
            data["collision"] = {"id": 1, "threat_level": 1, "delta": 0.0}
            pilot.tick()
            self.assertEqual(pilot.phase, Phase.FLYING)  # never enters BACKOFF
            self.assertIsNone(data.get("collision"))  # event consumed
        finally:
            pilot.shutdown()

    def test_backoff_levels_pitch_when_reverse_overspeed(self):
        """Reverse > BACKOFF_MAX_SPEED must not keep commanding hard nose-up."""
        from simulator.gp_pilot import BACKOFF_PITCH_DEG

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
        pilot._backoff_on = True  # backoff is opt-in now; test the mechanism
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
