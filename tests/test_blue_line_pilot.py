"""Unit tests for blue-line guidance (GP wireform, speed cap, turns, gate assist,
race-start gating)."""

import math
import os
import unittest

import numpy as np

from simulator.blue_line_pilot import (
    BlueLinePilot,
    CONF_GAIN_FLOOR,
    CORNER_SPEED_MPS,
    CRUISE_PITCH_DEG,
    CRUISE_SPEED_MPS,
    GATE_BIAS_ROLL_MAX_DEG,
    GATE_BIAS_YAW_MAX_DEG,
    GATE_STEER_BANK_MAX_DEG,
    GateAssist,
    HDG_MIN_BANDS,
    HOLD_YAW_ERR_DEG,
    HOVER_THRUST,
    MAX_SPEED_MPS,
    SEARCH_YAW_ERR_DEG,
    compute_blueline_guidance,
)
from simulator.gp_vision import YOLO_STALE_GAP_FR


class FakeController:
    def __init__(self):
        self.control_hz = None
        self.mode = None
        self.commands = []

    def set_control_mode(self, mode):
        self.mode = mode

    def set_attitude_quat_deg(self, roll, pitch, yaw, thrust):
        self.commands.append((roll, pitch, yaw, thrust))

    def arm(self):
        pass


def _race(sim_ms, start_ms=-1, finish_ns=-1):
    return {
        "sim_boot_time_ms": sim_ms,
        "race_start_boot_time_ms": start_ms,
        "race_finish_time_ns": finish_ns,
    }


def _make_pilot(data):
    ctrl = FakeController()
    pilot = BlueLinePilot(ctrl, data)
    pilot._open_log = lambda: None  # keep unit tests from writing rl/data files
    return pilot, ctrl


def _pose_data(bx, by, bz=0.0, pose_fid=10, cam_fid=None, conf=0.9):
    """Minimal shared-data view with one YOLO-pose gate detection."""
    return {
        "pose": {
            "frame_id": pose_fid,
            "infer_ms": 42.0,
            "gates": [
                {
                    "conf": conf,
                    "pose": {
                        "gate_pos_body": np.array([bx, by, bz]),
                        "normal_body": np.array([-1.0, 0.0, 0.0]),
                        "reproj_px": 1.0,
                    },
                }
            ],
        },
        "frame": {"frame_id": pose_fid if cam_fid is None else cam_fid},
    }


class GuidanceSignTests(unittest.TestCase):
    def test_right_offset_rolls_left_on_wire(self):
        # Corridor mid right → desired bank right → KR=-1 → negative wire roll.
        roll, pitch, yaw, thrust, dbg = compute_blueline_guidance(
            vision={
                "found": True,
                "cx_norm": 0.5,
                "cy_norm": 0.15,
                "heading_err": 0.0,
            },
            lost_s=0.0,
            roll_deg=0.0,
            pitch_deg=0.0,
        )
        self.assertEqual(dbg["mode"], "track")
        self.assertGreater(dbg["desired_roll"], 0.0)
        self.assertLess(roll, 0.0)
        self.assertLess(yaw, 0.0)
        self.assertAlmostEqual(pitch, CRUISE_PITCH_DEG, delta=3.0)
        self.assertGreater(thrust, 0.1)

    def test_left_offset_rolls_right_on_wire(self):
        roll, _, yaw, _, dbg = compute_blueline_guidance(
            vision={
                "found": True,
                "cx_norm": -0.5,
                "cy_norm": 0.15,
                "heading_err": 0.0,
            },
            lost_s=0.0,
            roll_deg=0.0,
            pitch_deg=0.0,
        )
        self.assertLess(dbg["desired_roll"], 0.0)
        self.assertGreater(roll, 0.0)
        self.assertGreater(yaw, 0.0)

    def test_low_in_image_dives(self):
        _, pitch_hi, _, thrust_hi, _ = compute_blueline_guidance(
            vision={
                "found": True,
                "cx_norm": 0.0,
                "cy_norm": 0.6,
                "heading_err": 0.0,
            },
            lost_s=0.0,
            pitch_deg=0.0,
        )
        _, pitch_lo, _, thrust_lo, _ = compute_blueline_guidance(
            vision={
                "found": True,
                "cx_norm": 0.0,
                "cy_norm": -0.4,
                "heading_err": 0.0,
            },
            lost_s=0.0,
            pitch_deg=0.0,
        )
        self.assertLess(pitch_hi, pitch_lo)
        self.assertLess(thrust_hi, thrust_lo)

    def test_hard_right_heading_yaws_and_banks(self):
        _, pitch_s, yaw_s, _, _ = compute_blueline_guidance(
            vision={
                "found": True,
                "cx_norm": 0.0,
                "cy_norm": 0.05,
                "heading_err": 0.0,
            },
            lost_s=0.0,
            pitch_deg=0.0,
        )
        roll_t, pitch_t, yaw_t, _, dbg = compute_blueline_guidance(
            vision={
                "found": True,
                "cx_norm": 0.1,
                "cy_norm": 0.05,
                "heading_err": 0.5,
            },
            lost_s=0.0,
            roll_deg=0.0,
            pitch_deg=0.0,
        )
        self.assertEqual(dbg["mode"], "track")
        self.assertLess(yaw_t, yaw_s)
        self.assertGreater(abs(yaw_t), abs(yaw_s))
        self.assertLess(roll_t, 0.0)  # wire: bank into right bend
        self.assertGreater(dbg["turn_mag"], 0.5)

    def test_corner_slows_v_target(self):
        _, _, _, _, dbg_s = compute_blueline_guidance(
            vision={
                "found": True,
                "cx_norm": 0.0,
                "cy_norm": 0.05,
                "heading_err": 0.0,
            },
            lost_s=0.0,
        )
        _, _, _, _, dbg_t = compute_blueline_guidance(
            vision={
                "found": True,
                "cx_norm": 0.2,
                "cy_norm": 0.05,
                "heading_err": 0.6,
            },
            lost_s=0.0,
        )
        self.assertLess(dbg_t["v_target"], dbg_s["v_target"])

    def test_over_max_speed_noses_up(self):
        state = {}
        _, pitch_fast, _, _, dbg = compute_blueline_guidance(
            vision={
                "found": True,
                "cx_norm": 0.0,
                "cy_norm": 0.05,
                "heading_err": 0.0,
            },
            lost_s=0.0,
            pitch_deg=0.0,
            vX=MAX_SPEED_MPS * 1.2,
            state=state,
        )
        self.assertGreater(dbg["pitch_des"], 0.0)
        self.assertGreater(pitch_fast, 0.0)

    def test_max_speed_is_25_kmh(self):
        self.assertAlmostEqual(MAX_SPEED_MPS * 3.6, 25.0, places=3)

    def test_search_right_hint(self):
        roll, _, yaw, _, dbg = compute_blueline_guidance(
            vision=None, lost_s=3.0, last_cx=0.0, last_hdg=0.4
        )
        self.assertEqual(dbg["mode"], "search")
        self.assertEqual(dbg["yaw_err"], SEARCH_YAW_ERR_DEG)
        self.assertEqual(yaw, -SEARCH_YAW_ERR_DEG)
        self.assertEqual(roll, 0.0)

    def test_search_left_hint(self):
        _, _, yaw, _, dbg = compute_blueline_guidance(
            vision=None, lost_s=3.0, last_cx=-0.5, last_hdg=0.0
        )
        self.assertEqual(dbg["mode"], "search")
        self.assertEqual(yaw, SEARCH_YAW_ERR_DEG)

    def test_hold_brief_loss_yaws_toward_hint(self):
        _, _, yaw, _, dbg = compute_blueline_guidance(
            vision=None, lost_s=0.1, last_hdg=0.3
        )
        self.assertEqual(dbg["mode"], "hold")
        self.assertEqual(yaw, -HOLD_YAW_ERR_DEG)

    def test_line_lost_fresh_gate_steers_toward_gate(self):
        # Gate 2 scenario: line gone, gate visible 18 deg LEFT -> yaw/bank left.
        roll, _, yaw, _, dbg = compute_blueline_guidance(
            vision=None,
            lost_s=1.0,
            gate={"bearing_deg": -18.0, "range_m": 6.0},
        )
        self.assertEqual(dbg["mode"], "gate")
        self.assertAlmostEqual(dbg["yaw_err"], -18.0, places=5)  # 1.0 * -18
        self.assertGreater(yaw, 0.0)  # KY=-1: left turn = positive wire yaw
        self.assertLess(dbg["desired_roll"], 0.0)
        self.assertGreaterEqual(dbg["desired_roll"], -GATE_STEER_BANK_MAX_DEG)
        self.assertGreater(roll, 0.0)  # KR=-1
        self.assertEqual(dbg["v_target"], CORNER_SPEED_MPS)

    def test_line_lost_no_gate_classic_search(self):
        _, _, yaw, _, dbg = compute_blueline_guidance(
            vision=None, lost_s=3.0, last_cx=0.0, last_hdg=0.4, gate=None
        )
        self.assertEqual(dbg["mode"], "search")
        self.assertEqual(dbg["yaw_err"], SEARCH_YAW_ERR_DEG)
        self.assertEqual(yaw, -SEARCH_YAW_ERR_DEG)

    def test_track_gate_bias_bounded(self):
        # Line tracked, bending left; gate 25 deg left agrees -> bounded extra
        # left command + earlier slow-down, never beyond the bias clamps.
        vis = {"found": True, "cx_norm": 0.0, "cy_norm": 0.05, "heading_err": -0.10}
        _, _, _, _, dbg_base = compute_blueline_guidance(vision=vis, lost_s=0.0)
        _, _, _, _, dbg_gate = compute_blueline_guidance(
            vision=vis,
            lost_s=0.0,
            gate={"bearing_deg": -25.0, "range_m": 8.0},
        )
        self.assertEqual(dbg_gate["mode"], "track")
        self.assertTrue(dbg_gate.get("gate_bias"))
        d_yaw = dbg_gate["yaw_err"] - dbg_base["yaw_err"]
        d_roll = dbg_gate["desired_roll"] - dbg_base["desired_roll"]
        self.assertLess(d_yaw, 0.0)
        self.assertLessEqual(abs(d_yaw), GATE_BIAS_YAW_MAX_DEG + 1e-6)
        self.assertLess(d_roll, 0.0)
        self.assertLessEqual(abs(d_roll), GATE_BIAS_ROLL_MAX_DEG + 1e-6)
        self.assertLess(dbg_gate["v_target"], dbg_base["v_target"])

    def test_track_gate_bias_skipped_on_disagreement(self):
        # Line clearly bends RIGHT while gate reads far LEFT: trust the line
        # for STEERING — but still slow down (direction-neutral corner brake).
        vis = {"found": True, "cx_norm": 0.0, "cy_norm": 0.05, "heading_err": 0.15}
        base = compute_blueline_guidance(vision=vis, lost_s=0.0)
        gated = compute_blueline_guidance(
            vision=vis,
            lost_s=0.0,
            gate={"bearing_deg": -25.0, "range_m": 8.0},
        )
        self.assertEqual(gated[4]["yaw_err"], base[4]["yaw_err"])
        self.assertEqual(gated[4]["desired_roll"], base[4]["desired_roll"])
        self.assertLess(gated[4]["v_target"], base[4]["v_target"])
        self.assertNotIn("gate_bias", gated[4])
        self.assertTrue(gated[4].get("gate_slow"))

    def test_bias_skipped_for_near_offpath_gate(self):
        # Straight line + big-bearing det at short range = wrong gate/phantom
        # (the log-verified +16..+42 deg dets that steered runs 1-2 RIGHT at
        # the left corner). Must not STEER — but slowing down is always safe.
        vis = {"found": True, "cx_norm": 0.0, "cy_norm": 0.05, "heading_err": 0.0}
        base = compute_blueline_guidance(vision=vis, lost_s=0.0)
        gated = compute_blueline_guidance(
            vision=vis,
            lost_s=0.0,
            gate={"bearing_deg": 30.0, "range_m": 12.0},
        )
        self.assertEqual(gated[4]["yaw_err"], base[4]["yaw_err"])
        self.assertEqual(gated[4]["desired_roll"], base[4]["desired_roll"])
        self.assertLess(gated[4]["v_target"], base[4]["v_target"])
        self.assertNotIn("gate_bias", gated[4])
        self.assertTrue(gated[4].get("gate_slow"))

    def test_bias_engages_for_far_onpath_gate(self):
        # Straight line + small-bearing FAR det = plausibly the next gate on
        # the course: anticipate the turn.
        vis = {"found": True, "cx_norm": 0.0, "cy_norm": 0.05, "heading_err": 0.0}
        base = compute_blueline_guidance(vision=vis, lost_s=0.0)
        gated = compute_blueline_guidance(
            vision=vis,
            lost_s=0.0,
            gate={"bearing_deg": -15.0, "range_m": 25.0},
        )
        self.assertTrue(gated[4].get("gate_bias"))
        self.assertLess(gated[4]["yaw_err"], base[4]["yaw_err"])
        self.assertLess(gated[4]["v_target"], base[4]["v_target"])

    def test_search_hint_sign_from_gate(self):
        # Line hint says right (last_hdg>0) but the last gate was LEFT: the
        # measured gate bearing must win the search direction.
        _, _, yaw, _, dbg = compute_blueline_guidance(
            vision=None,
            lost_s=3.0,
            last_cx=0.0,
            last_hdg=0.4,
            gate=None,
            gate_hint_sign=-1.0,
        )
        self.assertEqual(dbg["mode"], "search")
        self.assertEqual(dbg["yaw_err"], -SEARCH_YAW_ERR_DEG)
        self.assertEqual(yaw, SEARCH_YAW_ERR_DEG)

    def test_roll_feedback_levels_when_banked(self):
        # Already at desired bank → wire roll near 0.
        _, _, _, _, dbg = compute_blueline_guidance(
            vision={
                "found": True,
                "cx_norm": 0.4,
                "cy_norm": 0.05,
                "heading_err": 0.0,
            },
            lost_s=0.0,
            roll_deg=0.0,
        )
        desired = dbg["desired_roll"]
        roll_cmd, _, _, _, _ = compute_blueline_guidance(
            vision={
                "found": True,
                "cx_norm": 0.4,
                "cy_norm": 0.05,
                "heading_err": 0.0,
            },
            lost_s=0.0,
            roll_deg=desired,
        )
        self.assertAlmostEqual(roll_cmd, 0.0, places=5)


class ClimbAndWeaveTests(unittest.TestCase):
    """Baseline fine-tune: unstarve the gate-2 climb (turn_mag no longer cuts
    thrust so hard) + damp the cx weave with a bounded lateral-rate term."""

    def _low(self, cx=0.0, hdg=0.0, cy=-0.4):
        # Drone LOW in the corridor (cy negative) → needs to climb.
        return {"found": True, "cx_norm": cx, "cy_norm": cy, "heading_err": hdg}

    def test_climb_authority_restored_when_turning(self):
        # With turn_mag high AND the drone low, thrust must now climb harder
        # than the old 0.7-attenuation would have (35% -> 70% authority).
        from simulator.blue_line_pilot import HOVER_THRUST, K_THRUST_CY, CY_TARGET

        # A hard corner drives turn_mag to 1.0; drone is low → wants thrust up.
        vis = self._low(cx=0.0, hdg=0.6)  # big heading = real corner, tm→1
        _, _, _, thrust, dbg = compute_blueline_guidance(vision=vis, lost_s=0.0)
        self.assertGreater(dbg["turn_mag"], 0.9)
        cy_err = self._low()["cy_norm"] - CY_TARGET
        old = HOVER_THRUST - K_THRUST_CY * cy_err * (1.0 - 0.7 * dbg["turn_mag"])
        self.assertGreater(thrust, old)  # more climb than the old attenuation

    def test_straight_thrust_unchanged(self):
        # turn_mag→0 on a straight: the attenuation change is a no-op, so the
        # level-leg (gate 1) behavior is byte-identical.
        from simulator.blue_line_pilot import HOVER_THRUST, K_THRUST_CY, CY_TARGET

        vis = self._low(cx=0.0, hdg=0.0)
        _, _, _, thrust, dbg = compute_blueline_guidance(vision=vis, lost_s=0.0)
        self.assertAlmostEqual(dbg["turn_mag"], 0.0, places=6)
        cy_err = vis["cy_norm"] - CY_TARGET
        self.assertAlmostEqual(thrust, HOVER_THRUST - K_THRUST_CY * cy_err, places=6)

    def test_turn_mag_denoised(self):
        # Same cx weave, hdg dead: turn_mag is lower than the old 0.7 scale.
        vis = {"found": True, "cx_norm": 0.4, "cy_norm": 0.05, "heading_err": 0.0}
        _, _, _, _, dbg = compute_blueline_guidance(vision=vis, lost_s=0.0)
        self.assertAlmostEqual(dbg["turn_mag"], 0.4 / 1.1, places=3)
        self.assertLess(dbg["turn_mag"], 0.4 / 0.7)  # was the old value

    def test_cx_rate_damping_opposes_convergence(self):
        # cx falling (drone converging on the corridor) → D-term trims the
        # bank down so the correction doesn't overshoot into the weave.
        from simulator.blue_line_pilot import DCX_EMA_ALPHA, K_ROLL_DCX

        vis = {"found": True, "cx_norm": 0.3, "cy_norm": 0.05, "heading_err": 0.0}
        state = {"prev_cx": 0.5, "dcx_ema": 0.0}
        _, _, _, _, dbg = compute_blueline_guidance(
            vision=vis, lost_s=0.0, state=state, dt=0.1
        )
        _, _, _, _, dbg0 = compute_blueline_guidance(
            vision={"found": True, "cx_norm": 0.3, "cy_norm": 0.05, "heading_err": 0.0},
            lost_s=0.0,
        )
        self.assertLess(dbg["desired_roll"], dbg0["desired_roll"])
        expected_dcx = DCX_EMA_ALPHA * (0.3 - 0.5) / 0.1
        self.assertAlmostEqual(
            dbg["desired_roll"] - dbg0["desired_roll"], K_ROLL_DCX * expected_dcx, places=4
        )

    def test_cx_rate_contribution_capped(self):
        from simulator.blue_line_pilot import DCX_ROLL_MAX_DEG, K_ROLL_CX

        # A one-tick cx teleport must add at most the cap to the bank.
        vis = {"found": True, "cx_norm": 0.1, "cy_norm": 0.05, "heading_err": 0.0}
        state = {"prev_cx": -0.9, "dcx_ema": 0.0}
        _, _, _, _, dbg = compute_blueline_guidance(
            vision=vis, lost_s=0.0, state=state, dt=1.0 / 32.0
        )
        self.assertAlmostEqual(
            dbg["desired_roll"], K_ROLL_CX * 0.1 + DCX_ROLL_MAX_DEG, places=4
        )

    def test_loss_resets_cx_rate_state(self):
        state = {"prev_cx": 0.5, "dcx_ema": -1.0}
        compute_blueline_guidance(vision=None, lost_s=1.0, state=state)
        self.assertNotIn("prev_cx", state)
        self.assertEqual(state["dcx_ema"], 0.0)


class SpeedTuningTests(unittest.TestCase):
    @unittest.skipIf(
        bool(os.environ.get("BL_CRUISE_KMH")), "BL_CRUISE_KMH override active"
    )
    def test_cruise_default_is_12_kmh(self):
        self.assertAlmostEqual(CRUISE_SPEED_MPS * 3.6, 12.0, places=3)

    def test_gate_bearing_full_corner_brake(self):
        # An agreeing 25-deg gate bearing must pull v_target all the way down
        # to corner speed BEFORE the line itself bends in-frame.
        _, _, _, _, dbg = compute_blueline_guidance(
            vision={
                "found": True,
                "cx_norm": 0.0,
                "cy_norm": 0.05,
                "heading_err": -0.10,
            },
            lost_s=0.0,
            gate={"bearing_deg": -25.0, "range_m": 8.0},
        )
        self.assertEqual(dbg["mode"], "track")
        self.assertEqual(dbg["turn_mag"], 1.0)
        self.assertAlmostEqual(dbg["v_target"], CORNER_SPEED_MPS, places=6)


class TurnRateTests(unittest.TestCase):
    def test_slew_deg_s_param(self):
        from simulator.gp_pilot import CommandSlew

        fast = CommandSlew(hz=60.0, deg_s=240.0)
        fast.apply(0.0, 0.0, 0.0, 0.3)
        _, _, yaw, _ = fast.apply(0.0, 0.0, 25.0, 0.3)
        self.assertAlmostEqual(yaw, 4.0, places=6)  # 240/60 per tick
        default = CommandSlew(hz=60.0)
        default.apply(0.0, 0.0, 0.0, 0.3)
        _, _, yaw_d, _ = default.apply(0.0, 0.0, 25.0, 0.3)
        self.assertAlmostEqual(yaw_d, 1.5, places=6)  # GP unchanged: 90/60

    def test_turn_mag_latch_decays(self):
        # Corner flags must persist through one-frame vision flicker, then
        # decay — not whipsaw v_target 2.22<->3.33 tick to tick.
        # 0.75 rad = 43 deg: a genuinely hard corner. (Was 0.6 rad, which
        # saturated only at the pre-fix _TURN_HDG_SCALE_DEG of 20.)
        corner = {"found": True, "cx_norm": 0.0, "cy_norm": 0.05, "heading_err": 0.75}
        straight = {"found": True, "cx_norm": 0.0, "cy_norm": 0.05, "heading_err": 0.0}
        state = {}
        _, _, _, _, dbg1 = compute_blueline_guidance(
            vision=corner, lost_s=0.0, state=state
        )
        self.assertEqual(dbg1["turn_mag"], 1.0)
        _, _, _, _, dbg2 = compute_blueline_guidance(
            vision=straight, lost_s=0.0, state=state, dt=1.0 / 60.0
        )
        self.assertGreater(dbg2["turn_mag"], 0.95)  # one straight frame: held
        self.assertLess(dbg2["v_target"], CORNER_SPEED_MPS + 0.2)
        _, _, _, _, dbg3 = compute_blueline_guidance(
            vision=straight, lost_s=0.0, state=state, dt=1.0
        )
        self.assertLess(dbg3["turn_mag"], 0.3)  # a full second later: decayed

    def test_yaw_damping_opposes_rotation(self):
        # Sim gyros are sign-inverted: rotating RIGHT reads raw gz NEGATIVE.
        # Damping must cut a right command while the rotation builds.
        from simulator.blue_line_pilot import K_YAW_D_GZ

        vis = {"found": True, "cx_norm": 0.3, "cy_norm": 0.05, "heading_err": 0.0}
        _, _, _, _, d0 = compute_blueline_guidance(vision=vis, lost_s=0.0)
        _, _, _, _, d1 = compute_blueline_guidance(vision=vis, lost_s=0.0, gz_dps=-30.0)
        self.assertLess(d1["yaw_err"], d0["yaw_err"])
        self.assertAlmostEqual(
            d0["yaw_err"] - d1["yaw_err"], K_YAW_D_GZ * 30.0, places=5
        )
        # And rotating LEFT (raw gz positive) cuts a left command.
        _, _, _, _, d2 = compute_blueline_guidance(
            vision={
                "found": True,
                "cx_norm": -0.3,
                "cy_norm": 0.05,
                "heading_err": 0.0,
            },
            lost_s=0.0,
            gz_dps=30.0,
        )
        self.assertGreater(d2["yaw_err"], -0.3 * 30.0)  # magnitude reduced

    def test_openloop_corner_brake(self):
        # vX unknown (no velocity telemetry): full corner must pitch UP.
        _, pitch, _, _, dbg = compute_blueline_guidance(
            vision={
                "found": True,
                "cx_norm": 0.0,
                "cy_norm": 0.05,
                "heading_err": 0.6,
            },
            lost_s=0.0,
            pitch_deg=0.0,
        )
        self.assertGreater(dbg["pitch_des"], 0.0)
        self.assertGreater(pitch, 0.0)


class DetectorConfidenceTests(unittest.TestCase):
    """Per-band detector publishes conf + points; guidance must respect both."""

    def _vis(self, **kw):
        v = {"found": True, "cx_norm": 0.3, "cy_norm": 0.05, "heading_err": 0.0}
        v.update(kw)
        return v

    def test_missing_conf_behaves_as_full_confidence(self):
        # Callers predating the conf field must be unaffected.
        _, _, _, _, bare = compute_blueline_guidance(vision=self._vis(), lost_s=0.0)
        _, _, _, _, full = compute_blueline_guidance(
            vision=self._vis(conf=1.0), lost_s=0.0
        )
        self.assertAlmostEqual(bare["desired_roll"], full["desired_roll"])
        self.assertAlmostEqual(bare["yaw_err"], full["yaw_err"])

    def test_low_confidence_fades_but_never_kills_steering(self):
        _, _, _, _, hi = compute_blueline_guidance(
            vision=self._vis(conf=1.0), lost_s=0.0
        )
        _, _, _, _, lo = compute_blueline_guidance(
            vision=self._vis(conf=0.0), lost_s=0.0
        )
        self.assertLess(abs(lo["yaw_err"]), abs(hi["yaw_err"]))
        # Floor: a weak lock still steers the right way, at half authority.
        self.assertAlmostEqual(lo["yaw_err"], CONF_GAIN_FLOOR * hi["yaw_err"])
        self.assertGreater(lo["yaw_err"], 0.0)

    def test_short_line_fit_gates_heading_only(self):
        # Heading is a fit: too few bands and its slope means nothing. cx,
        # which needs no fit, must still steer.
        short = self._vis(heading_err=0.6, points=[(1.0, 2.0), (3.0, 4.0)])
        _, _, _, _, dbg = compute_blueline_guidance(vision=short, lost_s=0.0)
        self.assertEqual(dbg["hdg"], 0.0)
        self.assertGreater(dbg["yaw_err"], 0.0)  # cx still drives it

    def test_enough_bands_keeps_heading(self):
        pts = [(float(i), float(i)) for i in range(HDG_MIN_BANDS)]
        long_fit = self._vis(heading_err=0.6, points=pts)
        _, _, _, _, dbg = compute_blueline_guidance(vision=long_fit, lost_s=0.0)
        self.assertAlmostEqual(dbg["hdg"], 0.6)

    def test_gentle_bend_no_longer_forces_a_full_corner_brake(self):
        # Retuned _TURN_HDG_SCALE_DEG: heading now resolves every frame, so a
        # ~19 deg gentle bend must NOT latch turn_mag to a full brake.
        gentle = self._vis(cx_norm=0.0, heading_err=math.radians(19.0))
        _, _, _, _, dbg = compute_blueline_guidance(vision=gentle, lost_s=0.0)
        self.assertLess(dbg["turn_mag"], 0.8)
        self.assertGreater(dbg["v_target"], CORNER_SPEED_MPS)

    def test_hard_corner_still_brakes_fully(self):
        hard = self._vis(cx_norm=0.0, heading_err=math.radians(40.0))
        _, _, _, _, dbg = compute_blueline_guidance(vision=hard, lost_s=0.0)
        self.assertGreaterEqual(dbg["turn_mag"], 1.0)
        self.assertAlmostEqual(dbg["v_target"], CORNER_SPEED_MPS)


class RaceGateTests(unittest.TestCase):
    def _flying_pilot(self):
        """Drive a pilot to GO via a fresh race start."""
        data = {"armed": True, "race_status": _race(300)}
        pilot, ctrl = _make_pilot(data)
        pilot.tick()  # anchors at sim_ms=300, no race scheduled
        data["race_status"] = _race(600, start_ms=3289)
        pilot.tick()  # fresh race -> GO immediately (no countdown hold)
        assert pilot._phase == "fly"
        return pilot, ctrl, data

    def test_holds_zero_thrust_without_race(self):
        data = {"armed": True}
        pilot, ctrl = _make_pilot(data)
        for _ in range(3):
            pilot.tick()
        self.assertEqual(pilot._phase, "wait")
        self.assertEqual(ctrl.commands[-1], (0.0, 0.0, 0.0, 0.0))

    def test_stale_running_race_never_flies(self):
        # Race already running when the pilot came up: not our countdown.
        data = {"armed": True, "race_status": _race(60000, start_ms=5000)}
        pilot, ctrl = _make_pilot(data)
        pilot.tick()
        data["race_status"] = _race(60400, start_ms=5000)
        pilot.tick()
        self.assertEqual(pilot._phase, "wait")
        self.assertEqual(ctrl.commands[-1][3], 0.0)

    def test_fresh_race_start_flies_immediately(self):
        # No countdown hold: wheels-up the moment a fresh race start appears.
        data = {"armed": True, "race_status": _race(300)}
        pilot, ctrl = _make_pilot(data)
        pilot.tick()
        self.assertEqual(pilot._phase, "wait")
        data["race_status"] = _race(600, start_ms=3289)  # countdown still running
        pilot.tick()
        self.assertEqual(pilot._phase, "fly")
        self.assertEqual(ctrl.commands[-1][3], HOVER_THRUST)
        pilot.tick()  # guidance now runs (line lost -> hold/search, thrust > 0)
        self.assertGreater(ctrl.commands[-1][3], 0.0)

    def test_midflight_restart_returns_to_hold(self):
        pilot, ctrl, data = self._flying_pilot()
        data["race_status"] = _race(9000, start_ms=12000)  # Restart Race clicked
        pilot.tick()
        self.assertEqual(pilot._phase, "wait")
        self.assertEqual(ctrl.commands[-1], (0.0, 0.0, 0.0, 0.0))

    def test_race_finish_returns_to_hold(self):
        pilot, ctrl, data = self._flying_pilot()
        data["race_status"] = _race(20000, start_ms=3289, finish_ns=5)
        pilot.tick()
        self.assertEqual(pilot._phase, "wait")
        self.assertEqual(ctrl.commands[-1][3], 0.0)

    def test_sim_clock_rewind_returns_to_hold(self):
        pilot, ctrl, data = self._flying_pilot()
        data["race_status"] = _race(5000, start_ms=3289)  # race progresses
        pilot.tick()
        data["race_status"] = _race(200, start_ms=-1)  # in-sim reset rewinds boot
        pilot.tick()
        self.assertEqual(pilot._phase, "wait")
        self.assertEqual(ctrl.commands[-1][3], 0.0)

    def test_go_blocked_while_physics_frozen(self):
        # A stale race idles the physics: IMU streams but its clock freezes
        # and commands do nothing. GO must wait for a live clock.
        data = {
            "armed": True,
            "race_status": _race(300),
            "imu": {"time_us": 5, "gz": 0.0},
        }
        pilot, ctrl = _make_pilot(data)
        pilot.tick()
        data["race_status"] = _race(600, start_ms=3289)
        pilot.tick()  # fresh race, but first clock sighting: hold
        pilot.tick()  # clock static: still hold
        self.assertEqual(pilot._phase, "wait")
        self.assertEqual(ctrl.commands[-1][3], 0.0)
        data["imu"] = {"time_us": 6, "gz": 0.0}
        pilot.tick()  # clock advanced: physics live -> GO
        self.assertEqual(pilot._phase, "fly")


class GateAssistTests(unittest.TestCase):
    def test_assist_fresh_pose_yields_left_bearing(self):
        out = GateAssist().update(_pose_data(8.0, -3.0), now=0.0)
        self.assertIsNotNone(out)
        self.assertAlmostEqual(out["bearing_deg"], -20.556, places=2)
        self.assertAlmostEqual(out["range_m"], 8.544, places=2)
        self.assertEqual(out["lag_fr"], 0)
        self.assertEqual(out["infer_ms"], 42.0)

    def test_assist_stale_pose_is_none(self):
        # Inference fell behind the camera clock -> never steer on an old world.
        data = _pose_data(8.0, -3.0, pose_fid=10, cam_fid=10 + YOLO_STALE_GAP_FR + 1)
        self.assertIsNone(GateAssist().update(data, now=0.0))

    def test_assist_near_gate_suppressed(self):
        # Threading the gate (bx < 2 m): bearings blow up -> assist goes blind.
        self.assertIsNone(GateAssist().update(_pose_data(1.5, 0.2), now=0.0))

    def test_assist_rejects_far_range(self):
        # Gates are 24-29 m apart; the logged 58-120 m "detections" were pure
        # PnP noise and poisoned the turn direction.
        self.assertIsNone(GateAssist().update(_pose_data(60.0, -5.0), now=0.0))

    def test_assist_frozen_camera_is_none(self):
        data = _pose_data(8.0, -3.0)
        data["frame"]["received_at"] = 0.0
        self.assertIsNone(GateAssist().update(data, now=10.0))

    def test_assist_wide_bearing_rejected(self):
        # |bearing| beyond the half-FOV is a clip artifact, not guidance.
        self.assertIsNone(GateAssist().update(_pose_data(3.0, 4.0), now=0.0))


if __name__ == "__main__":
    unittest.main()
