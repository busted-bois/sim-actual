import os
import unittest

from simulator.pilot_profile import load_profile


class TestPilotProfile(unittest.TestCase):
    def tearDown(self):
        for key in ("PILOT_AB", "PILOT_DYNAMICS", "PILOT_JACOBIAN"):
            os.environ.pop(key, None)

    def test_preset_main(self):
        os.environ["PILOT_AB"] = "main"
        p = load_profile()
        self.assertEqual(p.label, "main")
        self.assertEqual(p.jacobian_blend, 0.0)
        self.assertFalse(p.use_angle_p)
        self.assertTrue(p.obstacle_avoidance)

    def test_preset_branch(self):
        os.environ["PILOT_AB"] = "branch"
        p = load_profile()
        self.assertEqual(p.label, "branch")
        self.assertEqual(p.jacobian_blend, 0.0)
        self.assertFalse(p.use_angle_p)
        self.assertTrue(p.telemetry_use_active_gate)

    def test_preset_jacobian(self):
        os.environ["PILOT_AB"] = "jacobian"
        p = load_profile()
        self.assertEqual(p.label, "main+jacobian")
        self.assertGreater(p.jacobian_blend, 0.0)
        self.assertFalse(p.use_angle_p)

    def test_branch_legacy(self):
        os.environ["PILOT_AB"] = "branch-legacy"
        p = load_profile()
        self.assertEqual(p.label, "branch-legacy")
        self.assertTrue(p.use_angle_p)
        self.assertAlmostEqual(p.hover_thrust, 0.27)

    def test_independent_flags(self):
        os.environ["PILOT_DYNAMICS"] = "main"
        os.environ["PILOT_JACOBIAN"] = "1"
        p = load_profile()
        self.assertEqual(p.label, "main+jacobian")
        self.assertFalse(p.use_angle_p)
        self.assertGreater(p.jacobian_blend, 0.0)


if __name__ == "__main__":
    unittest.main()
