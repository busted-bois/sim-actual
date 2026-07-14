"""Gate aim + PD-hook units for rl.fly2_course (no sim required)."""

import unittest

import numpy as np

from rl import spec
from rl.fly2_course import (
    HOVER_T,
    K_ATT,
    KD_Z,
    KP_Z,
    Fly2Config,
    gate_target_z,
    rates_from_attitude_targets,
)

SIGNS = (-1.0, 1.0, -1.0)  # explicit so a live-measured signs.json can't leak in


class GateTargetZTests(unittest.TestCase):
    def test_flipz_climb_course_aims_at_opening_centre(self):
        # Real rl/data/gate_map.json gate 1: base z=5.07 (flip convention),
        # h=2.72. Centre = -(5.07) - 1.36 = -6.43 NED — the old zoff=-1.0 law
        # aimed -6.07, ~0.4 m below the opening centre.
        g = {"pos": [-46.89, -2.50, 5.07], "h": 2.72}
        cfg = Fly2Config(flipz=True)
        self.assertAlmostEqual(gate_target_z(g, cfg), -6.43, places=2)

    def test_plain_ned_course_aims_at_opening_centre(self):
        g = {"pos": [10.0, 0.0, -5.07], "h": 2.72}
        cfg = Fly2Config(flipz=False)
        self.assertAlmostEqual(gate_target_z(g, cfg), -6.43, places=2)

    def test_ground_gate_targets_half_height_up(self):
        # Gate 0 sits at ground level (base z ~ 0): aim h/2 above the ground.
        g = {"pos": [-23.30, -0.40, -0.03], "h": 2.72}
        cfg = Fly2Config(flipz=True)
        self.assertAlmostEqual(gate_target_z(g, cfg), 0.03 - 1.36, places=2)

    def test_height_falls_back_to_spec(self):
        g = {"pos": [0.0, 0.0, -4.0]}
        cfg = Fly2Config()
        self.assertAlmostEqual(
            gate_target_z(g, cfg), -4.0 - spec.GATE_SIZE_M / 2.0, places=6
        )

    def test_zoff_is_a_trim_around_centre(self):
        g = {"pos": [0.0, 0.0, -4.0], "h": 2.72}
        self.assertAlmostEqual(
            gate_target_z(g, Fly2Config(zoff=-0.5)),
            gate_target_z(g, Fly2Config(zoff=0.0)) - 0.5,
            places=6,
        )


class PDHookTests(unittest.TestCase):
    def _expected_p_only(self, roll, pitch, z, vz, tr, tp, yerr, tz):
        s_r, s_p, s_y = SIGNS
        rr = float(np.clip(s_r * K_ATT * (tr - roll), -0.30, 0.30))
        pr = float(np.clip(s_p * K_ATT * (tp - pitch), -0.30, 0.30))
        yr = float(np.clip(s_y * 0.4 * yerr, -0.5, 0.5))
        th = float(np.clip(HOVER_T + KP_Z * (z - tz) + KD_Z * vz, 0.18, 0.5))
        return rr, pr, yr, th

    def test_zero_kd_matches_pure_p_law(self):
        args = (0.05, -0.03, -2.0, 0.1, 0.0, -0.08, 0.2, -6.43)
        got = rates_from_attitude_targets(
            *args, signs=SIGNS, k_d=0.0, gyro=(0.3, -0.2, 0.1)
        )
        self.assertEqual(got, self._expected_p_only(*args))

    def test_kd_damps_against_measured_rate(self):
        # Plant responds rate = sign*cmd, so damping a positive measured rate
        # requires sign*cmd < 0 on EVERY axis — including sign=-1 roll/yaw,
        # where the old (damping outside the sign) form commanded the opposite
        # and pumped energy in.
        damped = rates_from_attitude_targets(
            0.0,
            0.0,
            -2.0,
            0.0,
            0.0,
            0.0,
            0.0,
            -2.0,
            signs=SIGNS,
            k_d=0.15,
            gyro=(0.5, 0.5, 0.5),
        )
        for i, s in enumerate(SIGNS):
            self.assertLess(s * damped[i], 0.0, f"axis {i} must oppose +rate")


class Fly2MakeFlyStartupTests(unittest.TestCase):
    """Offline wiring for `make fly` (live TRAINING session still required)."""

    def test_takeoff_constants(self):
        import rl.fly2 as fly2

        self.assertEqual(fly2.TAKEOFF_S, 4.0)
        self.assertEqual(fly2.HOVER_Z_NED, -3.0)
        self.assertLessEqual(fly2.TAKEOFF_THRUST, 0.32)
        self.assertLess(fly2.TAKEOFF_IDLE_THRUST, fly2.HOVER_T)
        self.assertGreater(fly2.TAKEOFF_KD, 0.0)

    def test_tilt_rate_signs_flip_pitch(self):
        """IMU tilt path must invert signs.json pitch (+1 → −1)."""
        from rl.fly2 import _rate_signs
        from rl.fly2_course import rates_from_attitude_targets

        odo = (-1.0, 1.0, -1.0)
        self.assertEqual(_rate_signs("odo", odo), odo)
        self.assertEqual(_rate_signs("mav_attitude", odo), odo)
        tilt = _rate_signs("tilt", odo)
        self.assertEqual(tilt, (-1.0, -1.0, -1.0))
        # Nose-down → negative pitch_cmd with tilt signs (was +0.30 → dive).
        pcmd = rates_from_attitude_targets(
            0.0,
            -0.5,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            -3.0,
            signs=tilt,
            k_d=0.12,
            gyro=(0.0, -0.76, 0.0),
        )[1]
        self.assertLess(pcmd, 0.0)

    def test_default_skips_enter_wait(self):
        """make fly/fly-vision must not block on input(); race GO gates launch."""
        import inspect

        import rl.fly2 as fly2

        src = inspect.getsource(fly2.main)
        self.assertIn('--wait"', src)
        self.assertIn('action="store_true"', src)
        self.assertNotIn("store_false", src)
        self.assertNotIn("READY -- press ENTER the moment", src)

    def test_sim_interface_live_boundary(self):
        from rl.sim_interface import SimInterface

        self.assertTrue(callable(getattr(SimInterface, "ensure_armed", None)))
        self.assertTrue(callable(getattr(SimInterface, "send_attitude_rates", None)))


if __name__ == "__main__":
    unittest.main()
