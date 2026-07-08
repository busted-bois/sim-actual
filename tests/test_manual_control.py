"""Headless unit tests for the manual-flight control law (no sim required)."""

import math
import unittest

from simulator.manual_control import (
    ALTITUDE_TRIM,
    DEFAULT_SPEED_KMH,
    SPEED_STEP_KMH,
    ManualControl,
)


class FakeController:
    """Captures the last send_attitude_rates(...) call."""

    def __init__(self):
        self.last = None

    def send_attitude_rates(self, roll_rate, pitch_rate, yaw_rate, thrust):
        self.last = {
            "roll": roll_rate,
            "pitch": pitch_rate,
            "yaw": yaw_rate,
            "thrust": thrust,
        }


def pressed(*keys):
    """Return an is_pressed(key) callable reporting the given keys as held."""
    held = set(keys)
    return lambda key: key in held


def level_data(z=0.0, vz=0.0):
    """Odometry telemetry for a level drone at altitude z (no horizontal motion)."""
    return {"odometry": {"q": (1.0, 0.0, 0.0, 0.0), "z": z, "vz": vz}}


class TestManualControl(unittest.TestCase):
    def _mc(self, keys, data=None):
        ctl = FakeController()
        mc = ManualControl(ctl, data or level_data(), is_pressed=pressed(*keys))
        return ctl, mc

    def _tick(self, keys, data=None):
        ctl, mc = self._mc(keys, data)
        mc.tick()
        return ctl.last

    def test_no_keys_level_hovers(self):
        out = self._tick([])
        self.assertAlmostEqual(out["roll"], 0.0, places=6)
        self.assertAlmostEqual(out["pitch"], 0.0, places=6)
        self.assertAlmostEqual(out["yaw"], 0.0, places=6)
        # at zero altitude error the PID outputs its feedforward (hover) thrust
        self.assertAlmostEqual(out["thrust"], ALTITUDE_TRIM, places=6)

    def test_w_pitches_forward(self):
        # W wants forward speed -> nose-down (negative) pitch-rate command.
        out = self._tick(["w"])
        self.assertLess(out["pitch"], 0.0)

    def test_s_pitches_back(self):
        out = self._tick(["s"])
        self.assertGreater(out["pitch"], 0.0)

    def test_a_d_roll_opposite_signs(self):
        left = self._tick(["a"])["roll"]
        right = self._tick(["d"])["roll"]
        self.assertLess(left, 0.0)
        self.assertGreater(right, 0.0)
        self.assertAlmostEqual(left, -right, places=6)

    def test_t_climbs_x_descends(self):
        # T raises the altitude setpoint -> more thrust than hover; X the opposite.
        self.assertGreater(self._tick(["t"])["thrust"], ALTITUDE_TRIM)
        self.assertLess(self._tick(["x"])["thrust"], ALTITUDE_TRIM)

    def test_hover_trim_shifts_neutral_thrust(self):
        # '=' trims hover up, '-' trims it down; neutral thrust follows.
        base = self._tick([])["thrust"]
        self.assertGreater(self._tick(["equal"])["thrust"], base)
        self.assertLess(self._tick(["minus"])["thrust"], base)

    def test_q_e_yaw_opposite_signs(self):
        self.assertLess(self._tick(["q"])["yaw"], 0.0)
        self.assertGreater(self._tick(["e"])["yaw"], 0.0)

    def test_leveling_corrects_tilt(self):
        # Drone rolled right (+) with no key held -> command should roll back (-).
        data = {"odometry": {"q": _roll_quat(math.radians(5.0)), "z": 0.0, "vz": 0.0}}
        out = self._tick([], data=data)
        self.assertLess(out["roll"], 0.0)

    def test_altitude_hold_adds_thrust_when_low(self):
        # Below the latched hold altitude (NED z larger = lower) -> more thrust.
        ctl, mc = self._mc([])
        mc.tick()  # latches z_target = 0.0
        mc.data["odometry"]["z"] = 1.0  # drifted down 1 m
        mc.tick()
        self.assertGreater(ctl.last["thrust"], ALTITUDE_TRIM)

    def test_default_speed_is_5_kmh(self):
        _, mc = self._mc([])
        self.assertAlmostEqual(mc.speed_kmh, DEFAULT_SPEED_KMH, places=6)

    def test_r_increases_f_decreases_speed(self):
        _, up = self._mc(["r"])
        up.tick()
        self.assertAlmostEqual(up.speed_kmh, DEFAULT_SPEED_KMH + SPEED_STEP_KMH, 6)

        _, down = self._mc(["f"])
        down.tick()
        self.assertAlmostEqual(down.speed_kmh, DEFAULT_SPEED_KMH - SPEED_STEP_KMH, 6)

    def test_speed_step_is_edge_triggered(self):
        # Holding R across many ticks steps the speed only once.
        _, mc = self._mc(["r"])
        for _ in range(10):
            mc.tick()
        self.assertAlmostEqual(mc.speed_kmh, DEFAULT_SPEED_KMH + SPEED_STEP_KMH, 6)


def _roll_quat(roll):
    """Quaternion (w, x, y, z) for a pure roll rotation."""
    return (math.cos(roll / 2.0), math.sin(roll / 2.0), 0.0, 0.0)


if __name__ == "__main__":
    unittest.main()
