"""Offline tests for flightlab Bus altitude/position plumbing."""

import unittest
from unittest import mock

from flightlab.bus import Bus, _local_ned_from_data


class LocalNedHelperTests(unittest.TestCase):
    def test_from_local_position_ned_dict(self):
        data = {
            "local_position_ned": {
                "x": 1.0,
                "y": 2.0,
                "z": -3.0,
                "vx": 0.1,
                "vy": 0.0,
                "vz": -0.2,
            }
        }
        pos, vel = _local_ned_from_data(data)
        self.assertEqual(pos, (1.0, 2.0, -3.0))
        self.assertEqual(vel, (0.1, 0.0, -0.2))

    def test_from_legacy_pos_ned_tuple(self):
        data = {
            "has_position": True,
            "pos_ned": (0.5, -1.0, -2.5),
            "vel_ned": (0.0, 0.0, 0.0),
        }
        pos, vel = _local_ned_from_data(data)
        self.assertEqual(pos[2], -2.5)
        self.assertEqual(vel, (0.0, 0.0, 0.0))

    def test_missing_returns_none(self):
        self.assertIsNone(_local_ned_from_data({}))


class BusReadStateTests(unittest.TestCase):
    def _bare_bus(self) -> Bus:
        with mock.patch.object(Bus, "__init__", lambda self: None):
            bus = Bus()
        bus.data = {}
        bus._pose_mono = None
        bus._last_stamp = None
        bus.estimator = mock.MagicMock()
        bus.estimator.ready = False
        return bus

    def test_attitude_plus_pos_ned_sets_alt_trusted(self):
        bus = self._bare_bus()
        bus.data = {
            "armed": True,
            "attitude": {
                "roll": 0.0,
                "pitch": 0.1,
                "yaw": 0.2,
                "roll_speed": 0.0,
                "pitch_speed": 0.0,
                "yaw_speed": 0.0,
            },
            "has_position": True,
            "pos_ned": (0.0, 0.0, -0.5),
            "vel_ned": (0.0, 0.0, -0.1),
            "imu": {
                "ax": 0.0,
                "ay": 0.0,
                "az": -9.81,
                "gx": 0.0,
                "gy": 0.0,
                "gz": 0.0,
            },
        }
        s = bus._read_state()
        self.assertIsNotNone(s)
        assert s is not None
        self.assertEqual(s.pose_source, "attitude")
        self.assertTrue(s.alt_trusted)
        self.assertAlmostEqual(s.pos_ned[2], -0.5)
        self.assertAlmostEqual(s.vel_ned[2], -0.1)


if __name__ == "__main__":
    unittest.main()
