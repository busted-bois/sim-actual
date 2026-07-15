"""fly-policy gate-map resolve: live burst miss must fall back to JSON."""

from __future__ import annotations

import json
import os
import tempfile
import types
import unittest
from unittest import mock

import numpy as np

from rl.sim_interface import SimInterface
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
    def test_hover_action_maps_near_live_hover(self):
        from rl.deploy import LIVE_HOVER_THRUST, LIVE_RATE_CLIP, live_scale_action

        # Policy idle (all zeros) → train thrust 0.5 → live hover.
        cmd = live_scale_action(np.zeros(4))
        self.assertAlmostEqual(cmd[3], LIVE_HOVER_THRUST, places=5)
        self.assertTrue(np.all(np.abs(cmd[:3]) <= LIVE_RATE_CLIP + 1e-9))

    def test_saturated_policy_cannot_command_4rad_or_full_thrust(self):
        from rl.deploy import LIVE_RATE_CLIP, LIVE_THRUST_MAX, live_scale_action

        cmd = live_scale_action(np.ones(4))
        self.assertLessEqual(abs(cmd[0]), LIVE_RATE_CLIP + 1e-9)
        self.assertLessEqual(abs(cmd[1]), LIVE_RATE_CLIP + 1e-9)
        self.assertLessEqual(abs(cmd[2]), LIVE_RATE_CLIP + 1e-9)
        self.assertLessEqual(cmd[3], LIVE_THRUST_MAX + 1e-9)
        self.assertLess(cmd[3], 0.9)  # must not dump train-env "1.0" thrust live


if __name__ == "__main__":
    unittest.main()
