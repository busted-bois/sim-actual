"""Offline units for the flightlab attitude harness fixes (no sim required)."""

import json
import math
import os
import tempfile
import time
import unittest
from unittest import mock

from flightlab import metrics
from flightlab.controllers import PDController, _thrust
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
        # Below the 0.5 deg dead-band: 0.0 = no evidence (never a silent +1).
        self.assertEqual(identify_sign_from_pulse(0.0, 0.001, 0.2), 0.0)


class GyroDoubletSignTests(unittest.TestCase):
    def test_clean_positive_convention(self):
        # +pulse raises the rate, -pulse lowers it -> +1.
        self.assertEqual(metrics.sign_from_gyro_doublet(0.0, 0.18, -0.18, 0.2), 1.0)

    def test_clean_inverted_convention(self):
        self.assertEqual(metrics.sign_from_gyro_doublet(0.0, -0.18, 0.18, 0.2), -1.0)

    def test_weak_response_is_unknown(self):
        # Response below min_resp: 0.0, never a silent +1.
        self.assertEqual(metrics.sign_from_gyro_doublet(0.0, 0.02, -0.02, 0.2), 0.0)

    def test_disagreeing_halves_are_unknown(self):
        # Both halves moved the same way -> drift, not a command response.
        self.assertEqual(metrics.sign_from_gyro_doublet(0.0, 0.18, 0.18, 0.2), 0.0)

    def test_baseline_rotation_subtracted(self):
        # Drone already rotating at 0.4 rad/s: doublet still resolves vs base.
        self.assertEqual(metrics.sign_from_gyro_doublet(0.4, 0.58, 0.22, 0.2), 1.0)


class EulerDeltaSignTests(unittest.TestCase):
    def test_pure_drift_is_unknown(self):
        # The whole delta is explained by pre-pulse drift -> no evidence.
        drift = math.radians(3.4) / 0.3
        self.assertEqual(
            metrics.euler_delta_sign(0.0, math.radians(3.4), drift, 0.3, 0.2), 0.0
        )

    def test_real_delta_signs(self):
        self.assertEqual(
            metrics.euler_delta_sign(0.0, math.radians(3.0), 0.0, 0.3, 0.2), 1.0
        )
        self.assertEqual(
            metrics.euler_delta_sign(0.0, -math.radians(3.0), 0.0, 0.3, 0.2), -1.0
        )

    def test_yaw_wraps_across_pi(self):
        # 179 deg -> -178 deg is +3 deg once wrapped, not -357.
        self.assertEqual(
            metrics.euler_delta_sign(
                math.radians(179.0), math.radians(-178.0), 0.0, 0.3, 0.2
            ),
            1.0,
        )


class B4VerdictTests(unittest.TestCase):
    def test_failing_unity_gain_fails(self):
        # Regression: the old last_stable=1.0 default reported PASS while the
        # 1.0x sub-runs themselves failed.
        from flightlab.run_attitude import _b4_verdict

        last, onset, passed = _b4_verdict(
            [{"gain_scale": 1.0, "B2": False, "B3": False}]
        )
        self.assertEqual(last, 0.0)
        self.assertEqual(onset, 1.0)
        self.assertFalse(passed)

    def test_stable_then_onset(self):
        from flightlab.run_attitude import _b4_verdict

        last, onset, passed = _b4_verdict(
            [
                {"gain_scale": 1.0, "B2": True, "B3": True},
                {"gain_scale": 1.5, "B2": True, "B3": True},
                {"gain_scale": 2.0, "B2": False, "B3": True},
            ]
        )
        self.assertEqual(last, 1.5)
        self.assertEqual(onset, 2.0)
        self.assertTrue(passed)


class PDDampingSignTests(unittest.TestCase):
    def test_negative_sign_axes_damp_not_excite(self):
        # Plant responds rate = sign*cmd: damping a +rate needs sign*cmd < 0 on
        # every axis. The old form (kd outside the sign) pumped energy into
        # sign=-1 roll/yaw.
        c = PDController(
            k_att=0.0,
            k_d=0.15,
            signs={"roll": -1.0, "pitch": 1.0, "yaw": -1.0},
        )
        s = _state()
        s.gyro = (0.5, 0.5, 0.5)
        cmd = c.update(s, Target(z=0.0), 1.0 / 90.0)
        for u, sign in (
            (cmd.roll_rate, -1.0),
            (cmd.pitch_rate, 1.0),
            (cmd.yaw_rate, -1.0),
        ):
            self.assertLess(sign * u, 0.0)


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

    def test_ekf_alt_does_not_trip_low_floor(self):
        # EKF z can spike without a real climb — must not false-trip prep.
        m = self._expired_monitor()
        s_climb = _state(z=-1.0)
        s_climb.pose_source = "ekf"
        self.assertFalse(m.check(s_climb).tripped)
        s_ground = _state(z=-(ALT_LOW_M / 2))
        s_ground.pose_source = "ekf"
        self.assertFalse(m.check(s_ground).tripped)

    def test_reset_clears_airborne_latch(self):
        m = self._expired_monitor()
        m.check(_state(z=-1.0))  # latch airborne
        m.reset()
        m._arm_mono = time.monotonic() - 10.0
        self.assertFalse(m.check(_state(z=-0.03)).tripped)


class AltUntrustedTripTests(unittest.TestCase):
    def _expired_monitor(self):
        m = SafetyMonitor()
        m.reset()
        m._arm_mono = time.monotonic() - 10.0
        return m

    def test_trips_after_debounce(self):
        # No trusted alt and no EKF fallback -> the thrust law silently
        # freezes at hover; the monitor must make that loud within ~0.5 s.
        m = self._expired_monitor()
        t0 = time.monotonic()
        s = _state(z=-1.0, alt_trusted=False, t=t0)
        s.pose_source = "attitude"
        self.assertFalse(m.check(s).tripped)
        s2 = _state(z=-1.0, alt_trusted=False, t=t0 + 0.6)
        s2.pose_source = "attitude"
        r = m.check(s2)
        self.assertTrue(r.tripped)
        self.assertEqual(r.reason, "alt_untrusted")

    def test_ekf_pose_is_exempt(self):
        m = self._expired_monitor()
        t0 = time.monotonic()
        for dt in (0.0, 0.6, 1.2):
            s = _state(z=-1.0, alt_trusted=False, t=t0 + dt)
            s.pose_source = "ekf"
            self.assertFalse(m.check(s).tripped)

    def test_recovering_alt_resets_debounce(self):
        m = self._expired_monitor()
        t0 = time.monotonic()
        s = _state(z=-1.0, alt_trusted=False, t=t0)
        s.pose_source = "attitude"
        self.assertFalse(m.check(s).tripped)
        # Alt comes back before the debounce elapses -> timer resets.
        self.assertFalse(m.check(_state(z=-1.0, t=t0 + 0.3)).tripped)
        s3 = _state(z=-1.0, alt_trusted=False, t=t0 + 0.7)
        s3.pose_source = "attitude"
        self.assertFalse(m.check(s3).tripped)


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

    def test_thrust_freezes_when_unknown_pose_source(self):
        s = _state(z=0.0, alt_trusted=False)
        s.pose_source = "unknown"
        t = _thrust(s, Target(z=-3.0))
        self.assertEqual(t, HOVER_THRUST)


if __name__ == "__main__":
    unittest.main()
