"""Offline units for the flightlab attitude harness fixes (no sim required)."""

import json
import os
import tempfile
import time
import unittest
from unittest import mock

from flightlab import metrics
from flightlab.controllers import _thrust
from flightlab.metrics import identify_sign_from_pulse, load_signs
from flightlab.safety import ALT_LOW_M, HOVER_THRUST, SafetyMonitor
from flightlab.state import State, Target


def _state(z=0.0, vz=0.0, roll=0.0, pitch=0.0, alt_trusted=True, armed=True, t=None):
    return State(
        t_mono=time.monotonic() if t is None else t,
        armed=armed,
        roll=roll,
        pitch=pitch,
        yaw=0.0,
        roll_rate=0.0,
        pitch_rate=0.0,
        yaw_rate=0.0,
        pos_ned=(0.0, 0.0, z),
        vel_ned=(0.0, 0.0, vz),
        gyro=(0.0, 0.0, 0.0),
        quat=(1.0, 0.0, 0.0, 0.0),
        pose_age_s=0.0,
        alt_trusted=alt_trusted,
    )


class SignTests(unittest.TestCase):
    def test_load_signs_fallback_is_all_positive(self):
        # The old {-1,-1,-1} fallback made every axis positive feedback (drone
        # nosed into the gate base). Live-verified convention is all +1.
        with tempfile.TemporaryDirectory() as d:
            missing = os.path.join(d, "signs.json")
            with mock.patch.object(metrics, "SIGNS_PATH", missing):
                signs = load_signs()
        self.assertEqual(signs, {"roll": 1.0, "pitch": 1.0, "yaw": 1.0})

    def test_load_signs_prefers_measured_file(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "signs.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump({"roll": -1.0, "pitch": 1.0, "yaw": -1.0}, f)
            with mock.patch.object(metrics, "SIGNS_PATH", path):
                signs = load_signs()
        self.assertEqual(signs, {"roll": -1.0, "pitch": 1.0, "yaw": -1.0})

    def test_identify_sign_from_pulse(self):
        # +rate pulse moved the angle + -> convention is +1; moved - -> -1.
        self.assertEqual(identify_sign_from_pulse(0.0, 0.10, 0.2), 1.0)
        self.assertEqual(identify_sign_from_pulse(0.0, -0.10, 0.2), -1.0)
        # Below the 0.5 deg dead-band: default +1 (no evidence).
        self.assertEqual(identify_sign_from_pulse(0.0, 0.001, 0.2), 1.0)


class SafetyTakeoffLatchTests(unittest.TestCase):
    def _expired_monitor(self):
        """Monitor whose 3 s arm-grace has already expired."""
        m = SafetyMonitor()
        m.reset()
        m._arm_mono = time.monotonic() - 10.0
        return m

    def test_no_low_alt_trip_on_ground_before_takeoff(self):
        # Spawn is on the ground (alt ~ 0 < ALT_LOW_M): must NOT trip before
        # the drone has ever been airborne, even with the grace expired.
        m = self._expired_monitor()
        r = m.check(_state(z=-0.03))  # alt = 0.03 m
        self.assertFalse(r.tripped, r.reason)

    def test_low_alt_trips_after_climb_then_ground(self):
        m = self._expired_monitor()
        self.assertFalse(m.check(_state(z=-1.0)).tripped)  # climbed to 1 m
        r = m.check(_state(z=-(ALT_LOW_M / 2)))  # back near the ground
        self.assertTrue(r.tripped)
        self.assertEqual(r.reason, "alt<0.2m")

    def test_reset_clears_airborne_latch(self):
        m = self._expired_monitor()
        m.check(_state(z=-1.0))  # latch airborne
        m.reset()
        m._arm_mono = time.monotonic() - 10.0
        self.assertFalse(m.check(_state(z=-0.03)).tripped)


class ThrustLawTests(unittest.TestCase):
    def test_thrust_closes_loop_when_alt_trusted(self):
        # 3 m below target -> more than hover thrust.
        t = _thrust(_state(z=0.0), Target(z=-3.0))
        self.assertGreater(t, HOVER_THRUST)

    def test_thrust_freezes_at_hover_when_alt_untrusted(self):
        # Documented fallback: without trusted altitude the law holds hover
        # thrust (prep now fails fast instead of relying on this in Training).
        t = _thrust(_state(z=0.0, alt_trusted=False), Target(z=-3.0))
        self.assertEqual(t, HOVER_THRUST)


if __name__ == "__main__":
    unittest.main()
