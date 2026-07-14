"""Offline unit checks for tilt filter + sim pose source preference."""

import unittest
from unittest.mock import MagicMock

from simulator.tilt_filter import TiltFilter


class TiltFilterTests(unittest.TestCase):
    def test_gyro_integrates_pitch(self):
        t = TiltFilter()
        t.update(
            {
                "ax": 0.0,
                "ay": 0.0,
                "az": -9.81,
                "gx": 0.0,
                "gy": 0.5,
                "gz": 0.0,
                "time_us": 0,
            },
            allow_accel=False,
        )
        t.update(
            {
                "ax": 0.0,
                "ay": 0.0,
                "az": -9.81,
                "gx": 0.0,
                "gy": 0.5,
                "gz": 0.0,
                "time_us": 100_000,
            },
            allow_accel=False,
        )
        self.assertAlmostEqual(t.pitch, 0.05, places=3)

    def test_accel_blend_blocked_when_disallowed(self):
        t = TiltFilter()
        # Seed near level via accel.
        for i, us in enumerate((0, 10_000)):
            t.update(
                {
                    "ax": 0.0,
                    "ay": 0.0,
                    "az": -9.81,
                    "gx": 0.0,
                    "gy": 0.0,
                    "gz": 0.0,
                    "time_us": us,
                },
                allow_accel=True,
                boost=True,
            )
        self.assertTrue(t.sync_to_accel())
        pitched = t.pitch
        # Hard forward accel would look like a pitch tip if blended.
        t.update(
            {
                "ax": 8.0,
                "ay": 0.0,
                "az": -6.0,
                "gx": 0.0,
                "gy": 0.0,
                "gz": 0.0,
                "time_us": 20_000,
            },
            allow_accel=False,
        )
        self.assertEqual(t.pitch, pitched)


class SnapshotPrefersAttitudeTests(unittest.TestCase):
    def test_attitude_not_overwritten_by_ekf(self):
        from rl.sim_interface import SimInterface

        sim = object.__new__(SimInterface)
        sim.use_estimator = False
        sim._shadow = None
        sim._shadow_next_t = 0.0
        sim.data = {
            "armed": True,
            "attitude": {
                "roll": 0.1,
                "pitch": -0.3,
                "yaw": 0.0,
                "roll_speed": 0.0,
                "pitch_speed": 0.0,
                "yaw_speed": 0.0,
            },
            "has_position": True,
            "pos_ned": (1.0, 2.0, -3.0),
            "vel_ned": (0.0, 0.0, 0.0),
            "yaw_rad": 0.0,
            "imu": None,
            "frame": None,
            "gates": [],
        }
        est = MagicMock()
        est.ready = True
        # Divergent EKF pose (would have caused takeoff pitch runaway if preferred).
        est.pose.return_value = (
            (9.0, 0.0, 0.8),
            (9.0, 0.0, 0.0),
            (0.7071, 0.0, 0.7071, 0.0),  # ~90° pitch
        )
        sim.estimator = est
        snap = SimInterface.snapshot(sim)
        self.assertIsNotNone(snap.quat)
        # Attitude path: pitch ~-0.3 rad, not the EKF 90° tip.
        from rl.fly2_course import rpy

        _r, pitch, _y = rpy(snap.quat)
        self.assertLess(abs(pitch - (-0.3)), 0.05)
        self.assertEqual(snap.pos_ned, (1.0, 2.0, -3.0))


if __name__ == "__main__":
    unittest.main()
