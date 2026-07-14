"""RateCmdShaper / GateCarrot / course-rate smoothing units (no sim required)."""

import math
import unittest
from unittest import mock

from rl import fly2_course
from rl.fly2_course import (
    HOVER_T,
    Fly2Config,
    GateCarrot,
    RateCmdShaper,
    compute_course_rates,
    gate_target_z,
)

# Real rl/data/gate_map.json gates 0-1 (flip convention, climb course).
GATES = [
    {"pos": [-23.30, -0.40, -0.03], "h": 2.72},
    {"pos": [-46.89, -2.50, 5.07], "h": 2.72},
]
DT = 1.0 / 90.0


class FakeClock:
    def __init__(self, t: float = 100.0):
        self.t = t

    def __call__(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += dt


class ClockedCase(unittest.TestCase):
    """Patches fly2_course's monotonic clock so dt is deterministic."""

    def setUp(self):
        self.clock = FakeClock()
        patcher = mock.patch.object(fly2_course.time, "monotonic", self.clock)
        patcher.start()
        self.addCleanup(patcher.stop)


class RateCmdShaperTests(ClockedCase):
    def test_first_thrust_passes_through(self):
        # Slewing thrust up from 0 would delay takeoff by ~0.5 s.
        sh = RateCmdShaper()
        self.assertEqual(sh.apply(0.0, 0.0, 0.0, 0.30)[3], 0.30)

    def test_rate_step_is_slew_limited(self):
        sh = RateCmdShaper(tau_s=0.0, slew=2.0)  # tau=0 isolates the slew
        sh.apply(0.0, 0.0, 0.0, HOVER_T)
        self.clock.advance(DT)
        r, p, y, _ = sh.apply(0.30, -0.30, 0.50, HOVER_T)
        lim = 2.0 * DT
        self.assertAlmostEqual(r, lim, places=9)
        self.assertAlmostEqual(p, -lim, places=9)
        self.assertAlmostEqual(y, lim, places=9)

    def test_thrust_step_is_slew_limited(self):
        sh = RateCmdShaper(thrust_slew=0.6)
        sh.apply(0.0, 0.0, 0.0, HOVER_T)
        self.clock.advance(DT)
        th = sh.apply(0.0, 0.0, 0.0, 0.40)[3]
        self.assertAlmostEqual(th, HOVER_T + 0.6 * DT, places=9)

    def test_converges_to_held_command(self):
        sh = RateCmdShaper()
        sh.apply(0.0, 0.0, 0.0, HOVER_T)
        out = (0.0, 0.0, 0.0, 0.0)
        for _ in range(200):
            self.clock.advance(DT)
            out = sh.apply(0.25, 0.0, 0.0, 0.35)
        self.assertAlmostEqual(out[0], 0.25, places=3)
        self.assertAlmostEqual(out[3], 0.35, places=6)

    def test_reset_clears_state(self):
        sh = RateCmdShaper()
        for _ in range(50):
            self.clock.advance(DT)
            sh.apply(0.30, 0.30, 0.30, 0.40)
        sh.reset()
        self.clock.advance(DT)
        self.assertEqual(sh.apply(0.0, 0.0, 0.0, HOVER_T)[3], HOVER_T)

    def test_stalled_caller_dt_is_clamped(self):
        # A 2 s stall must not let one tick jump the full command range.
        sh = RateCmdShaper(tau_s=0.0, slew=2.0)
        sh.apply(0.0, 0.0, 0.0, HOVER_T)
        self.clock.advance(2.0)
        r = sh.apply(0.30, 0.0, 0.0, HOVER_T)[0]
        self.assertLessEqual(r, 2.0 * (1.0 / 30.0) + 1e-9)


class GateCarrotTests(ClockedCase):
    def setUp(self):
        super().setUp()
        self.cfg = Fly2Config(flipz=True)

    def test_first_update_snaps_to_gate(self):
        c = GateCarrot()
        ax, ay, az = c.update(0, GATES, self.cfg)
        self.assertEqual((ax, ay), (-23.30, -0.40))
        self.assertAlmostEqual(az, gate_target_z(GATES[0], self.cfg), places=9)

    def test_transition_is_rate_bounded_and_arrives(self):
        c = GateCarrot(v_xy=6.0, v_z=2.0)
        prev = c.update(0, GATES, self.cfg)
        tz1 = gate_target_z(GATES[1], self.cfg)
        for _ in range(500):
            self.clock.advance(DT)
            cur = c.update(1, GATES, self.cfg)
            self.assertLessEqual(
                math.hypot(cur[0] - prev[0], cur[1] - prev[1]), 6.0 * DT + 1e-9
            )
            self.assertLessEqual(abs(cur[2] - prev[2]), 2.0 * DT + 1e-9)
            prev = cur
        # ~23.7 m of xy at 6 m/s and ~5.1 m of z at 2 m/s both fit in 500 ticks.
        self.assertEqual((prev[0], prev[1]), (-46.89, -2.50))
        self.assertAlmostEqual(prev[2], tz1, places=9)

    def test_reset_snaps_to_next_target(self):
        c = GateCarrot()
        c.update(0, GATES, self.cfg)
        c.reset()
        ax, ay, _ = c.update(1, GATES, self.cfg)
        self.assertEqual((ax, ay), (-46.89, -2.50))


class ComputeCourseRatesTests(ClockedCase):
    POSE = dict(
        pos_ned=(-10.0, 0.0, -3.0),
        vel_ned=(1.0, 0.0, 0.0),
        quat=(1.0, 0.0, 0.0, 0.0),
        hold_z=-3.0,
    )

    def _rates(self, **kw):
        args = dict(self.POSE)
        args.update(kw)
        return compute_course_rates(
            args["pos_ned"],
            args["vel_ned"],
            args["quat"],
            args.get("active", 0),
            GATES,
            args["hold_z"],
            args.get("cfg", Fly2Config(flipz=True)),
            gyro=args.get("gyro", (0.0, 0.0, 0.0)),
            carrot=args.get("carrot"),
            k_d=args.get("k_d"),
        )

    def test_carrot_none_matches_fresh_carrot_first_tick(self):
        # A fresh carrot snaps to the gate, so tick 1 must equal legacy aim.
        self.assertEqual(self._rates(), self._rates(carrot=GateCarrot()))

    def test_course_complete_ignores_carrot(self):
        c = GateCarrot()
        got = self._rates(active=len(GATES), carrot=c)
        self.assertEqual(got, (0.0, 0.0, 0.0, HOVER_T))
        self.assertIsNone(c._aim)  # never consulted past the last gate

    def test_kd_passthrough_changes_rates(self):
        a = self._rates(gyro=(0.5, 0.5, 0.5), k_d=0.0)
        b = self._rates(gyro=(0.5, 0.5, 0.5), k_d=0.2)
        self.assertNotEqual(a[:3], b[:3])


class DefaultKdTests(unittest.TestCase):
    def _reset_cache(self):
        fly2_course._measured_k_d_cache = "unset"

    def test_reads_measured_calibration(self):
        self._reset_cache()
        self.addCleanup(self._reset_cache)
        with mock.patch.object(
            fly2_course, "load_calibration", return_value={"k_d": 0.12}
        ):
            self.assertEqual(fly2_course._default_k_d(), 0.12)

    def test_falls_back_to_module_constant(self):
        self._reset_cache()
        self.addCleanup(self._reset_cache)
        with mock.patch.object(fly2_course, "load_calibration", return_value={}):
            self.assertEqual(fly2_course._default_k_d(), fly2_course.K_D)


if __name__ == "__main__":
    unittest.main()
