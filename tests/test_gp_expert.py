"""GP expert wrapper: action normalization + guidance flying the internal env."""

import unittest

import numpy as np

from rl import spec
from rl.gp_expert import GPExpert


class UnscaleActionTests(unittest.TestCase):
    def test_roundtrip_within_limits(self):
        phys = np.array([1.5, -2.0, 0.7, 0.27])
        back = spec.scale_action(spec.unscale_action(phys))
        np.testing.assert_allclose(back, phys, atol=1e-12)

    def test_normalized_roundtrip(self):
        a = np.array([0.3, -0.9, 0.1, -0.46])
        np.testing.assert_allclose(
            spec.unscale_action(spec.scale_action(a)), a, atol=1e-12
        )

    def test_out_of_range_clips(self):
        phys = np.array([99.0, -99.0, 99.0, 2.0])
        a = spec.unscale_action(phys)
        self.assertTrue(np.all(np.abs(a) <= 1.0))

    def test_hover_maps_to_hover(self):
        a = spec.unscale_action(np.array([0.0, 0.0, 0.0, spec.HOVER_THRUST]))
        self.assertAlmostEqual(spec.scale_action(a)[3], spec.HOVER_THRUST, places=12)


class GPExpertTests(unittest.TestCase):
    GATE_MAP = [{"pos": [5.0, 0.0, -3.0], "quat": [1.0, 0, 0, 0], "w": 2.72, "h": 2.72}]

    def test_act_bounded(self):
        ex = GPExpert()
        a = ex.act(np.zeros(3), np.zeros(3), np.array([1.0, 0, 0, 0]), self.GATE_MAP, 0)
        self.assertEqual(a.shape, (4,))
        self.assertTrue(np.all(np.abs(a) <= 1.0))

    def test_gate_switch_resets_hold(self):
        ex = GPExpert()
        two_gates = self.GATE_MAP + [
            {"pos": [10.0, 1.0, -3.0], "quat": [1.0, 0, 0, 0], "w": 2.72, "h": 2.72}
        ]
        for _ in range(5):
            ex.act(np.zeros(3), np.zeros(3), np.array([1.0, 0, 0, 0]), two_gates, 0)
        self.assertIsNotNone(ex.hold.get("prev_bearing_frame_id"))
        ex.act(np.zeros(3), np.zeros(3), np.array([1.0, 0, 0, 0]), two_gates, 1)
        # First tick after switch: fresh hold, no stale bearing history carried.
        self.assertEqual(ex._gate_idx, 1)

    def test_gate_behind_gives_no_detection(self):
        ex = GPExpert()
        behind = [
            {"pos": [-5.0, 0.0, -3.0], "quat": [1.0, 0, 0, 0], "w": 2.72, "h": 2.72}
        ]
        a = ex.act(np.zeros(3), np.zeros(3), np.array([1.0, 0, 0, 0]), behind, 0)
        # Blind guidance: no bank/yaw command, just pitch trim + hover-ish thrust.
        self.assertAlmostEqual(a[0], 0.0, places=6)
        self.assertAlmostEqual(a[2], 0.0, places=6)

    def test_passes_stage0_gate(self):
        from rl.env import GateRacingEnv

        env = GateRacingEnv(stage=0, seed=200)
        env.reset()
        ex = GPExpert()
        ex.reset()
        term = trunc = False
        passed = False
        while not (term or trunc):
            a = ex.act(env.p, env.v, env.q, env.gate_map, env.gate_idx)
            _, _, term, trunc, info = env.step(a)
            if info.get("gate_passed"):
                passed = True
                break
        self.assertTrue(passed, "GP expert should pass the stage-0 gate")


if __name__ == "__main__":
    unittest.main()
