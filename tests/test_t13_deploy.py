"""T13 focused tests: config, controlled prediction, PnP coast/reset, fallback, thrust."""

from __future__ import annotations

import math
import os
import tempfile
import types
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import torch
from numpy.testing import assert_allclose

from rl.estimation.ekf import ESKF, GRAVITY, C_THRUST
from rl.experts.vision_fallback import FallbackBrain, guidance_to_rate_cmds
from rl.experts.gp_expert import ATT_RATE_GAIN
from rl.training.train_ppo import StandalonePolicy, NET_ARCH
from simulator.vq2_pose import VQ2PoseEstimator


def _level_q():
    return np.array([1.0, 0.0, 0.0, 0.0])


def _hover_thrust():
    return GRAVITY / C_THRUST


def _good_det(gate_body, conf=0.9, range_m=5.0, reproj=1.0):
    return {
        "conf": conf,
        "pose": {
            "gate_pos_body": list(gate_body),
            "range_m": range_m,
            "reproj_px": reproj,
        },
    }


class TestVQ2ControlledPrediction(unittest.TestCase):
    """Raw powered accel must NOT drive velocity; commanded thrust must."""

    def test_zero_thrust_invokes_zupt(self):
        est = VQ2PoseEstimator()
        est.reset([{"pos": [0.0, 0.0, -5.0]}])
        data = {
            "imu": {
                "time_us": 10000,
                "ax": 0.0,
                "ay": 0.0,
                "az": 0.0,
                "gx": 0.0,
                "gy": 0.0,
                "gz": 0.0,
            },
        }
        est.tick(data, [{"pos": [10.0, 0.0, -5.0]}])
        assert_allclose(est.ekf.v, 0, atol=1e-15)

    def test_default_thrust_is_zero_zupt(self):
        est = VQ2PoseEstimator()
        est.reset([{"pos": [0.0, 0.0, -5.0]}])
        self.assertEqual(est.last_applied_thrust, 0.0)
        data = {
            "imu": {
                "time_us": 10000,
                "ax": 9.81,
                "ay": 0.0,
                "az": 0.0,
                "gx": 0.0,
                "gy": 0.0,
                "gz": 0.0,
            },
        }
        est.tick(data, [{"pos": [10.0, 0.0, -5.0]}])
        assert_allclose(est.ekf.v, 0, atol=1e-15)

    def test_hover_thrust_predicts_stable(self):
        est = VQ2PoseEstimator()
        gate_map = [{"pos": [0.0, 0.0, -5.0]}]
        est.reset(gate_map)
        p0 = est.ekf.p.copy()
        for i in range(500):
            data = {
                "imu": {
                    "time_us": i * 10000,
                    "ax": 0.0,
                    "ay": 0.0,
                    "az": 0.0,
                    "gx": 0.0,
                    "gy": 0.0,
                    "gz": 0.0,
                },
            }
            est.tick(data, gate_map, thrust_cmd=_hover_thrust())
        drift = np.linalg.norm(est.ekf.p - p0)
        self.assertLess(drift, 1.0)

    def test_raw_accel_ignored_under_power(self):
        est = VQ2PoseEstimator()
        est.reset([{"pos": [0.0, 0.0, -5.0]}])
        rng = np.random.default_rng(42)
        for i in range(200):
            garbage_accel = rng.uniform(-400, 400, 3)
            data = {
                "imu": {
                    "time_us": i * 10000,
                    "ax": garbage_accel[0],
                    "ay": garbage_accel[1],
                    "az": garbage_accel[2],
                    "gx": rng.normal(0, 0.002),
                    "gy": rng.normal(0, 0.002),
                    "gz": rng.normal(0, 0.002),
                },
            }
            est.tick(data, [{"pos": [10.0, 0.0, -5.0]}], thrust_cmd=_hover_thrust())
        self.assertTrue(est.ekf.healthy())

    def test_thrust_cmd_kwarg_sets_thrust(self):
        est = VQ2PoseEstimator()
        est.reset([{"pos": [0.0, 0.0, -5.0]}])
        data = {
            "imu": {
                "time_us": 10000,
                "ax": 0,
                "ay": 0,
                "az": 0,
                "gx": 0,
                "gy": 0,
                "gz": 0,
            }
        }
        est.tick(data, [{"pos": [10.0, 0.0, -5.0]}], thrust_cmd=0.35)
        self.assertAlmostEqual(est.last_applied_thrust, 0.35)

    def test_malformed_thrust_degrades_to_zero(self):
        est = VQ2PoseEstimator()
        est.reset([{"pos": [0.0, 0.0, -5.0]}])
        data = {
            "imu": {
                "time_us": 10000,
                "ax": 0,
                "ay": 0,
                "az": 0,
                "gx": 0,
                "gy": 0,
                "gz": 0,
            }
        }
        est.tick(data, [{"pos": [10.0, 0.0, -5.0]}], thrust_cmd=float("nan"))
        self.assertEqual(est.last_applied_thrust, 0.0)
        est.tick(data, [{"pos": [10.0, 0.0, -5.0]}], thrust_cmd=-1.0)
        self.assertEqual(est.last_applied_thrust, 0.0)
        est.tick(data, [{"pos": [10.0, 0.0, -5.0]}], thrust_cmd="bad")
        self.assertEqual(est.last_applied_thrust, 0.0)


class TestVQ2PnpFusion(unittest.TestCase):
    """Native data['pose'] PnP uses step_pnp_fusion coast/reset via tick."""

    def test_pnp_miss_coasts(self):
        est = VQ2PoseEstimator()
        gate_map = [{"pos": [10.0, 0.0, -5.0]}]
        est.reset(gate_map)
        coast_before = est.ekf.coast_count
        data = {"pose": {"frame_id": 1, "gates": []}, "active_gate_index": 0}
        est.tick(data, gate_map)
        self.assertGreater(est.ekf.coast_count, coast_before)

    def test_pnp_update_resets_coast(self):
        est = VQ2PoseEstimator()
        gate_map = [{"pos": [10.0, 0.0, -5.0]}]
        est.reset(gate_map)
        est.ekf.p = np.array([9.5, 0.0, -7.5])
        est.ekf.v = np.zeros(3)
        for fid in range(1, 4):
            data = {"pose": {"frame_id": fid, "gates": []}, "active_gate_index": 0}
            data["imu"] = {
                "time_us": fid * 10000,
                "ax": 0,
                "ay": 0,
                "az": 0,
                "gx": 0,
                "gy": 0,
                "gz": 0,
            }
            est.tick(data, gate_map, thrust_cmd=_hover_thrust())
        self.assertGreater(est.ekf.coast_count, 0)
        det = _good_det(np.array([0.5, 0.0, 2.5]), conf=0.95, range_m=3.0, reproj=0.5)
        data = {
            "pose": {"frame_id": 4, "gates": [det]},
            "active_gate_index": 0,
            "frame": {"frame_id": 4},
        }
        data["imu"] = {
            "time_us": 4 * 10000,
            "ax": 0,
            "ay": 0,
            "az": 0,
            "gx": 0,
            "gy": 0,
            "gz": 0,
        }
        est.tick(data, gate_map, thrust_cmd=_hover_thrust())
        self.assertEqual(est.ekf.coast_count, 0)

    def test_pnp_frame_dedup(self):
        est = VQ2PoseEstimator()
        gate_map = [{"pos": [10.0, 0.0, -5.0]}]
        est.reset(gate_map)
        det = _good_det(np.array([13.0, 0.0, 0.0]))
        data = {"pose": {"frame_id": 7, "gates": [det]}, "active_gate_index": 0}
        est.tick(data, gate_map)
        p_first = est.ekf.p.copy()
        est.tick(data, gate_map)
        assert_allclose(est.ekf.p, p_first)

    def test_pnp_hard_reset_after_prolonged_coast(self):
        est = VQ2PoseEstimator()
        gate_map = [{"pos": [10.0, 0.0, -5.0]}]
        est.reset(gate_map)
        for fid in range(1, 31):
            data = {"pose": {"frame_id": fid, "gates": []}, "active_gate_index": 0}
            est.tick(data, gate_map)
        self.assertEqual(est.ekf.coast_count, 30)
        det = _good_det(np.array([10.0, 0.0, 5.0]))
        data = {"pose": {"frame_id": 31, "gates": [det]}, "active_gate_index": 0}
        est.tick(data, gate_map)
        self.assertTrue(est.ekf.healthy())
        self.assertGreater(est.ekf.recovery_count, 0)


class TestDeployFallback(unittest.TestCase):
    """Forced divergence fallback + PnP-based recovery hysteresis."""

    def _make_brain_and_snap(self):
        def act(obs):
            return np.zeros(4)

        meta = {"train_hover": 0.27, "action_scale": (0.6, 0.6, 0.6)}
        t = [0.0]

        def now_fn():
            t[0] += 0.01
            return t[0]

        from rl.deploy import DeployBrain

        brain = DeployBrain(act, meta, now_fn=now_fn)
        gate_map = [
            {"pos": [-10.0, 0.0, -5.0], "quat": [1, 0, 0, 0], "w": 2.72, "h": 2.72}
        ]
        snap = types.SimpleNamespace(
            armed=True,
            pos_ned=None,
            vel_ned=None,
            quat=None,
            ang_vel=None,
            imu=None,
        )
        snap.has_pose = lambda: False
        brain.ekf = ESKF(p0=np.zeros(3), v0=np.zeros(3), q0=_level_q())
        brain.gate_idx = 0
        brain._last_imu_t = None
        brain._last_applied_thrust = 0.0
        brain._prev_signed = -1.0
        brain._armed_prev = True
        brain._in_fallback = False
        brain._accepted_pnp_frames = 0
        brain._last_gate_idx = 0
        brain._last_pnp_frame_id = None
        brain._last_gate_progress_t = None
        return brain, gate_map, snap

    def test_fallback_engages_on_high_covariance(self):
        brain, gate_map, snap = self._make_brain_and_snap()
        brain.ekf.P[:] = 1e6
        brain.tick(snap, gate_map, data={})
        self.assertTrue(brain._in_fallback)

    def test_seed_clears_stale_covariance_and_coast(self):
        brain, gate_map, snap = self._make_brain_and_snap()
        brain.ekf.P[:] = 1e6
        brain.ekf.coast(5, 0.01)
        brain._seed_ekf(snap, gate_map)
        assert_allclose(brain.ekf.P, np.eye(9))
        self.assertEqual(brain.ekf.coast_count, 0)
        self.assertTrue(brain._check_ekf_healthy())

    def test_fallback_engages_on_coast(self):
        brain, gate_map, snap = self._make_brain_and_snap()
        brain.ekf.coast(40, 0.01)
        brain.tick(snap, gate_map, data={})
        self.assertTrue(brain._in_fallback)

    def test_no_resume_without_fresh_pnp(self):
        brain, gate_map, snap = self._make_brain_and_snap()
        brain.ekf.P[:] = 1e6
        brain.tick(snap, gate_map, data={})
        self.assertTrue(brain._in_fallback)
        brain.ekf.reset(p0=np.zeros(3), v0=np.zeros(3))
        for _ in range(200):
            brain.tick(snap, gate_map, data={})
            if not brain._in_fallback:
                break
        self.assertTrue(brain._in_fallback, "should NOT resume without fresh PnP")

    def test_resume_after_accepted_pnp(self):
        from rl.deploy import RECOVERY_PNP_FRAMES

        brain, gate_map, snap = self._make_brain_and_snap()
        brain.ekf.P[:] = 1e6
        brain.tick(snap, gate_map, data={})
        self.assertTrue(brain._in_fallback)
        brain.ekf.reset(p0=np.array([-11.0, 0.0, -5.0]), v0=np.zeros(3))
        for i in range(RECOVERY_PNP_FRAMES + 1):
            pnp_data = {
                "pose": {
                    "frame_id": 1000 + i,
                    "gates": [
                        {
                            "conf": 0.9,
                            "pose": {
                                "gate_pos_body": [1.0, 0.0, 0.0],
                                "range_m": 1.0,
                                "reproj_px": 0.5,
                            },
                        }
                    ],
                },
                "active_gate_index": 0,
                "frame": {"frame_id": 1000 + i},
            }
            brain.tick(snap, gate_map, data=pnp_data)
            if not brain._in_fallback:
                break
        self.assertFalse(brain._in_fallback, "should resume after accepted PnP")


class TestMissingGateNet(unittest.TestCase):
    """FallbackBrain must work without gatenet.pt — no GateNet import."""

    def test_no_gatenet_import_required(self):
        import importlib

        fb_source = importlib.import_module("rl.experts.vision_fallback")
        src = fb_source.__file__
        with open(src) as f:
            src_text = f.read()
        self.assertNotIn("gatenet", src_text.lower())

    def test_no_gatenet_no_crash(self):
        fb = FallbackBrain()
        q = np.array([1.0, 0.0, 0.0, 0.0])
        roll, pitch, yaw, thrust = fb.update({}, q, 0.01)
        from rl.experts.vision_fallback import LIVE_HOVER_THRUST

        self.assertEqual((roll, pitch, yaw), (0.0, 0.0, 0.0))
        self.assertEqual(thrust, LIVE_HOVER_THRUST)

    @patch("rl.experts.vision_fallback.compute_guidance")
    def test_no_vision_does_not_issue_open_loop_guidance(self, compute):
        fb = FallbackBrain()
        cmd = fb.update({}, _level_q(), 0.01)
        self.assertEqual(cmd, (0.0, 0.0, 0.0, 0.27))
        compute.assert_not_called()


class TestGuidanceToRate(unittest.TestCase):
    """Degree commands -> rate convention conversion mirrors GPExpert."""

    def test_zero_is_zero(self):
        r, p, y, t = guidance_to_rate_cmds(0.0, 0.0, 0.0, 0.27)
        self.assertAlmostEqual(r, 0.0)
        self.assertAlmostEqual(p, 0.0)
        self.assertAlmostEqual(y, 0.0)
        self.assertAlmostEqual(t, 0.27)

    def test_signs_and_gain_match_gpexpert(self):
        deg2rad = math.pi / 180.0
        g = ATT_RATE_GAIN
        small = 2.0
        roll, pitch, yaw, _ = guidance_to_rate_cmds(small, small, -small, 0.3)
        expected_roll = -small * deg2rad * g
        expected_pitch = small * deg2rad * g
        expected_yaw = -(-small) * deg2rad * g
        self.assertAlmostEqual(roll, expected_roll, places=5)
        self.assertAlmostEqual(pitch, expected_pitch, places=5)
        self.assertAlmostEqual(yaw, expected_yaw, places=5)

    def test_clips_to_live_rate_clip(self):
        from rl.experts.vision_fallback import LIVE_RATE_CLIP

        huge = 90.0
        r, p, y, _ = guidance_to_rate_cmds(huge, huge, huge, 0.3)
        self.assertLessEqual(abs(r), LIVE_RATE_CLIP + 1e-9)
        self.assertLessEqual(abs(p), LIVE_RATE_CLIP + 1e-9)
        self.assertLessEqual(abs(y), LIVE_RATE_CLIP + 1e-9)


class TestFallbackVelocity(unittest.TestCase):
    """Fallback passes zero degraded translational velocity priors."""

    def test_vy_vd_zero_vx_nan(self):
        fb = FallbackBrain()
        q = np.array([1.0, 0.0, 0.0, 0.0])
        vision = {
            "frame_id": 1,
            "body_x_m": 5.0,
            "body_y_m": 0.0,
            "body_z_m": 0.0,
        }
        fb.smoother.update = MagicMock(return_value=vision)
        with patch("rl.experts.vision_fallback.compute_guidance") as mock_cg:
            mock_cg.return_value = (0.0, 0.0, 0.0, 0.27, {})
            fb.update({}, q, 0.01)
            _, kwargs = mock_cg.call_args
            self.assertAlmostEqual(kwargs["vY"], 0.0)
            self.assertAlmostEqual(kwargs["vD"], 0.0)
            self.assertTrue(math.isnan(kwargs["vX"]))


class TestConfigCLI(unittest.TestCase):
    """CLI config/policy/device overrides reach load_policy."""

    def test_load_policy_custom_path(self):
        std = StandalonePolicy()
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "custom.pt")
            torch.save({"state_dict": std.state_dict(), "arch": NET_ARCH}, path)
            from rl.deploy import load_policy

            act, meta = load_policy(path)
            self.assertIsNotNone(act)

    def test_config_loading(self):
        from rl.core.config import load_config, RLConfig
        from rl.deploy import DEFAULT_CONFIG_PATH

        cfg = load_config(DEFAULT_CONFIG_PATH)
        self.assertIsInstance(cfg, RLConfig)
        self.assertEqual(cfg.policy_path, "rl/data/policy.pt")
        self.assertEqual(cfg.device, "auto")

    @patch("rl.deploy.PolicyRunner")
    @patch("rl.deploy.resolve_device", return_value=torch.device("cpu"))
    @patch("rl.deploy.load_config")
    def test_cli_policy_and_device_override(self, load_cfg, resolve, runner):
        load_cfg.return_value = types.SimpleNamespace(
            policy_path="from-config.pt", device="auto"
        )
        from rl.deploy import main

        main(
            [
                "--config",
                "custom.yaml",
                "--policy",
                "override.pt",
                "--device",
                "cpu",
            ]
        )
        load_cfg.assert_called_once_with("custom.yaml")
        resolve.assert_called_once_with("cpu")
        runner.assert_called_once_with(policy_path="override.pt", device="cpu")
        runner.return_value.run.assert_called_once_with()


class TestGateNormalization(unittest.TestCase):
    def test_flat_course_no_flip(self):
        from rl.deploy import normalize_gate_map

        gm = [{"pos": [0.0, 0.0, 0.0]}, {"pos": [10.0, 0.0, -2.0]}]
        normed = normalize_gate_map(gm)
        self.assertEqual(len(normed), 2)
        self.assertAlmostEqual(normed[0]["pos"][2], 0.0)
        self.assertAlmostEqual(normed[1]["pos"][2], -2.0)
        gm[0]["pos"][2] = 99.0
        self.assertAlmostEqual(normed[0]["pos"][2], 0.0)

    def test_climb_course_flips(self):
        from rl.deploy import normalize_gate_map

        gm = [{"pos": [0.0, 0.0, 0.0]}, {"pos": [10.0, 0.0, 5.0]}]
        normed = normalize_gate_map(gm)
        self.assertAlmostEqual(normed[0]["pos"][2], 0.0)
        self.assertAlmostEqual(normed[1]["pos"][2], -5.0)

    def test_empty_map(self):
        from rl.deploy import normalize_gate_map

        self.assertEqual(normalize_gate_map([]), [])


class TestDeployRemapUnchanged(unittest.TestCase):
    def test_live_scale_action_math_unchanged(self):
        from rl.deploy import LIVE_HOVER_THRUST, live_scale_action

        meta = {"train_hover": 0.27, "action_scale": (0.6, 0.6, 0.6)}
        hover_a = np.array([0.0, 0.0, 0.0, 2.0 * 0.27 - 1.0])
        cmd = live_scale_action(hover_a, meta)
        self.assertAlmostEqual(cmd[3], LIVE_HOVER_THRUST, places=5)

    def test_legacy_remap_unchanged(self):
        from rl.deploy import (
            LEGACY_ACTION_SCALE,
            LEGACY_TRAIN_HOVER,
            LIVE_HOVER_THRUST,
            live_scale_action,
        )

        meta = {"train_hover": LEGACY_TRAIN_HOVER, "action_scale": LEGACY_ACTION_SCALE}
        cmd = live_scale_action(np.zeros(4), meta)
        self.assertAlmostEqual(cmd[3], LIVE_HOVER_THRUST, places=5)


if __name__ == "__main__":
    unittest.main()
