"""fly-policy gate-map resolve: live burst miss must fall back to JSON."""

from __future__ import annotations

import json
import os
import tempfile
import types
import unittest
from unittest import mock

import numpy as np

from rl.environment.sim_interface import SimInterface
from simulator.vision_rx import VisionRX


class CaptureGateMapFallbackTests(unittest.TestCase):
    def _stub(self):
        stub = types.SimpleNamespace(data={}, gate_list=lambda: [])
        stub.capture_gate_map = types.MethodType(SimInterface.capture_gate_map, stub)
        return stub

    def test_falls_back_to_saved_json_when_live_burst_missing(self):
        stub = self._stub()
        gates = [
            {
                "id": 0,
                "pos": [-23.0, -0.4, -0.03],
                "quat": [1.0, 0.0, 0.0, 0.0],
                "w": 2.72,
                "h": 2.72,
            }
        ]
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "gate_map.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump({"gates": gates}, f)
            out = stub.capture_gate_map(path=path, timeout_s=0.3)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["pos"][0], -23.0)

    def test_skips_nulled_vq2_burst_and_uses_saved(self):
        stub = self._stub()
        stub.data["track_positions_valid"] = False
        stub.gate_list = lambda: [
            {
                "id": 0,
                "pos": [0.0, 0.0, 0.0],
                "quat": [1.0, 0, 0, 0],
                "w": 0.0,
                "h": 0.0,
            }
        ]
        gates = [
            {
                "id": 0,
                "pos": [-10.0, 0.0, -1.0],
                "quat": [1.0, 0, 0, 0],
                "w": 2.72,
                "h": 2.72,
            }
        ]
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "gate_map.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump({"gates": gates}, f)
            out = stub.capture_gate_map(path=path, timeout_s=0.3)
        self.assertEqual(out[0]["pos"][0], -10.0)

    def test_no_saved_and_no_live_returns_empty(self):
        stub = self._stub()
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "missing.json")
            out = stub.capture_gate_map(path=path, timeout_s=0.2)
        self.assertEqual(out, [])


class QuietVisionTests(unittest.TestCase):
    def test_quiet_vision_suppresses_per_frame_spam(self):
        data = {"_quiet_vision": True}
        rx = VisionRX.__new__(VisionRX)
        rx.data = data
        rx._gate_was_detected = False
        with mock.patch("builtins.print") as mock_print:
            rx._log_gate_detected(
                True, cx=1.0, cy=2.0, area=100.0, nx=0.0, ny=0.0, range_m=5.0
            )
            rx._log_gate_detected(
                True, cx=1.0, cy=2.0, area=100.0, nx=0.0, ny=0.0, range_m=5.0
            )
        # Edge-only log once ("GATE acquired"), not per-frame spam.
        texts = " ".join(str(c) for c in mock_print.call_args_list)
        self.assertIn("GATE acquired", texts)
        self.assertNotIn("GATE cx=", texts)


class LiveRemapTests(unittest.TestCase):
    # Legacy = pre-calibration checkpoint (no metadata): hover 0.5, ±4/4/3.
    # Calibrated = post-calibration checkpoint: hover 0.27, ±0.6 caps.
    LEGACY = {"train_hover": 0.5, "action_scale": (4.0, 4.0, 3.0)}
    CALIBRATED = {"train_hover": 0.27, "action_scale": (0.6, 0.6, 0.6)}

    def test_legacy_hover_action_maps_to_live_hover(self):
        from rl.deploy import LIVE_HOVER_THRUST, LIVE_RATE_CLIP, live_scale_action

        # Legacy policy idle (all zeros) → train thrust 0.5 → live hover.
        cmd = live_scale_action(np.zeros(4), self.LEGACY)
        self.assertAlmostEqual(cmd[3], LIVE_HOVER_THRUST, places=5)
        self.assertTrue(np.all(np.abs(cmd[:3]) <= LIVE_RATE_CLIP + 1e-9))

    def test_calibrated_hover_action_maps_to_live_hover(self):
        from rl.deploy import LIVE_HOVER_THRUST, live_scale_action

        # Calibrated policy hovers at its OWN train hover (0.27), i.e.
        # a3 = 2*0.27-1 — this is the case the old 0.5-anchored remap sank.
        a = np.array([0.0, 0.0, 0.0, 2.0 * 0.27 - 1.0])
        cmd = live_scale_action(a, self.CALIBRATED)
        self.assertAlmostEqual(cmd[3], LIVE_HOVER_THRUST, places=5)

    def test_calibrated_thrust_is_identity_within_clips(self):
        from rl.deploy import LIVE_THRUST_MAX, LIVE_THRUST_MIN, live_scale_action

        # train hover == live hover → thrust passes through (then live clips).
        for t in (0.05, 0.2, 0.27, 0.4, 0.8):
            a = np.array([0.0, 0.0, 0.0, 2.0 * t - 1.0])
            cmd = live_scale_action(a, self.CALIBRATED)
            self.assertAlmostEqual(
                cmd[3], float(np.clip(t, LIVE_THRUST_MIN, LIVE_THRUST_MAX)), places=6
            )

    def test_rates_use_ckpt_scale_then_live_clip(self):
        from rl.deploy import LIVE_RATE_CLIP, live_scale_action

        # Calibrated ckpt half-stick = 0.5*0.6 = 0.3 rad/s — inside the clip,
        # must NOT be re-scaled by the current spec caps.
        cmd = live_scale_action(np.array([0.5, -0.5, 0.5, 0.0]), self.CALIBRATED)
        self.assertAlmostEqual(cmd[0], 0.30, places=6)
        self.assertAlmostEqual(cmd[1], -0.30, places=6)
        self.assertAlmostEqual(cmd[2], 0.30, places=6)
        # Legacy ckpt half-stick = 2 rad/s → live clip.
        cmd = live_scale_action(np.array([0.5, -0.5, 0.5, 0.0]), self.LEGACY)
        self.assertAlmostEqual(cmd[0], LIVE_RATE_CLIP, places=6)
        self.assertAlmostEqual(cmd[1], -LIVE_RATE_CLIP, places=6)

    def test_saturated_policy_cannot_command_4rad_or_full_thrust(self):
        from rl.deploy import LIVE_RATE_CLIP, LIVE_THRUST_MAX, live_scale_action

        for meta in (self.LEGACY, self.CALIBRATED):
            cmd = live_scale_action(np.ones(4), meta)
            self.assertLessEqual(abs(cmd[0]), LIVE_RATE_CLIP + 1e-9)
            self.assertLessEqual(abs(cmd[1]), LIVE_RATE_CLIP + 1e-9)
            self.assertLessEqual(abs(cmd[2]), LIVE_RATE_CLIP + 1e-9)
            self.assertLessEqual(cmd[3], LIVE_THRUST_MAX + 1e-9)
            self.assertLess(cmd[3], 0.9)  # never train-env "1.0" thrust live

    def test_legacy_checkpoint_loads_with_fallback_meta(self):
        import torch

        from rl.deploy import LEGACY_ACTION_SCALE, LEGACY_TRAIN_HOVER, load_policy
        from rl.training.train_ppo import NET_ARCH, StandalonePolicy

        std = StandalonePolicy()
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "policy.pt")
            torch.save(
                {
                    "state_dict": std.state_dict(),
                    "arch": NET_ARCH,
                },
                path,
            )
            act, meta = load_policy(path)
        self.assertEqual(meta["train_hover"], LEGACY_TRAIN_HOVER)
        self.assertEqual(meta["action_scale"], tuple(LEGACY_ACTION_SCALE))

    def test_missing_policy_exits_with_instructions(self):
        from rl.deploy import load_policy

        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(SystemExit) as ctx:
                load_policy(os.path.join(d, "nope.pt"))
        self.assertIn("train-ppo", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
