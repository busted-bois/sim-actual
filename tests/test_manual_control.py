"""Headless unit tests for the manual-flight control law (no sim required)."""

import math
import unittest

from simulator.controller import CONTROL_HZ
from simulator.manual_control import (
    CONTROL_DT_S,
    CRUISE_SPEED_KMH,
    HOVER_T,
    LAND_SETTLE_TICKS,
    ManualControl,
)


class FakeController:
    """Captures the last send_attitude_rates(...) call and counts arm() calls."""

    def __init__(self):
        self.last = None
        self.arm_calls = 0
        self.disarm_calls = 0

    def send_attitude_rates(self, roll_rate, pitch_rate, yaw_rate, thrust):
        self.last = {
            "roll": roll_rate,
            "pitch": pitch_rate,
            "yaw": yaw_rate,
            "thrust": thrust,
        }

    def arm(self):
        self.arm_calls += 1

    def disarm(self):
        self.disarm_calls += 1


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
        # at zero altitude error the PD outputs its feedforward (hover) thrust
        self.assertAlmostEqual(out["thrust"], HOVER_T, places=6)

    def test_w_pitches_forward(self):
        # W wants forward speed -> nose-down (negative) pitch-rate command.
        out = self._tick(["w"])
        self.assertLess(out["pitch"], 0.0)

    def test_s_pitches_back(self):
        out = self._tick(["s"])
        self.assertGreater(out["pitch"], 0.0)

    def test_a_d_roll_opposite_signs(self):
        # Wire roll command is inverted vs odometry attitude (SIGN_ROLL = -1,
        # measured by fly2_course): A (roll left) sends +, D (roll right) sends -.
        left = self._tick(["a"])["roll"]
        right = self._tick(["d"])["roll"]
        self.assertGreater(left, 0.0)
        self.assertLess(right, 0.0)
        self.assertAlmostEqual(left, -right, places=6)

    def test_r_climbs_f_descends(self):
        # R raises the altitude setpoint -> more thrust than hover; F the opposite.
        self.assertGreater(self._tick(["r"])["thrust"], HOVER_T)
        self.assertLess(self._tick(["f"])["thrust"], HOVER_T)

    def test_r_f_work_without_pose_telemetry(self):
        # Pose-blocked sessions stream no odometry (z/vz absent). R/F must STILL
        # command a climb/descend thrust open-loop, not sit at hover — the old code
        # gated them behind a latched z_target, so they did nothing (unnoticeable
        # in-sim while W/S kept working open-loop). data has no odometry -> z=None.
        up = self._tick(["r"], data={"armed": True})
        down = self._tick(["f"], data={"armed": True})
        self.assertGreater(up["thrust"], HOVER_T)
        self.assertLess(down["thrust"], HOVER_T)

    def test_hover_trim_shifts_neutral_thrust(self):
        # '=' trims hover up, '-' trims it down; neutral thrust follows.
        base = self._tick([])["thrust"]
        self.assertGreater(self._tick(["equal"])["thrust"], base)
        self.assertLess(self._tick(["minus"])["thrust"], base)

    def test_q_e_yaw_opposite_signs(self):
        # Q and E yaw in opposite directions. SIGN_YAW = +1 (flipped from the old
        # -1) so Q now sends negative and E positive — the direction the pilot
        # expects after the reported Q/E reversal.
        self.assertLess(self._tick(["q"])["yaw"], 0.0)
        self.assertGreater(self._tick(["e"])["yaw"], 0.0)

    def test_leveling_corrects_tilt(self):
        # Drone rolled right (+ in odometry) with no key held: under the measured
        # inverted-roll wire convention the corrective command is POSITIVE. This
        # locks in SIGN_ROLL = -1 — the +1 version made leveling positive feedback.
        data = {"odometry": {"q": _roll_quat(math.radians(5.0)), "z": 0.0, "vz": 0.0}}
        out = self._tick([], data=data)
        self.assertGreater(out["roll"], 0.0)

    def test_altitude_hold_adds_thrust_when_low(self):
        # Below the latched hold altitude (NED z larger = lower) -> more thrust.
        ctl, mc = self._mc([])
        mc.tick()  # latches z_target = 0.0
        mc.data["odometry"]["z"] = 1.0  # drifted down 1 m
        mc.tick()
        self.assertGreater(ctl.last["thrust"], HOVER_T)

    def test_cruise_speed_is_5_kmh(self):
        _, mc = self._mc([])
        self.assertAlmostEqual(mc.speed_kmh, CRUISE_SPEED_KMH, places=6)
        self.assertAlmostEqual(CRUISE_SPEED_KMH, 5.0, places=6)

    def test_rearms_while_disarmed(self):
        # A one-shot ARM at client start gets lost (sent before the sim
        # registers us) or undone (sim disarms after a crash) — measured live
        # 2026-07-08: HUD stuck at armed=no forever. tick() must re-arm.
        ctl, mc = self._mc([], data=dict(level_data(), armed=False))
        mc.tick()
        self.assertEqual(ctl.arm_calls, 1)
        mc.tick()  # immediately after: throttled, no spam
        self.assertEqual(ctl.arm_calls, 1)
        mc._next_arm_t = 0.0  # retry window elapsed
        mc.tick()
        self.assertEqual(ctl.arm_calls, 2)

    def test_rearms_when_armed_state_unknown(self):
        # No HEARTBEAT processed yet (armed key absent) — keep trying.
        ctl, mc = self._mc([])
        mc.tick()
        self.assertEqual(ctl.arm_calls, 1)

    def test_does_not_arm_when_armed(self):
        ctl, mc = self._mc([], data=dict(level_data(), armed=True))
        mc.tick()
        mc._next_arm_t = 0.0
        mc.tick()
        self.assertEqual(ctl.arm_calls, 0)

    def test_status_flags_pose_blocked_when_imu_only(self):
        # Event/qualification sessions stream HIGHRES_IMU but block
        # ODOMETRY/ATTITUDE/LOCAL_POSITION_NED (measured via make probe
        # 2026-07-08). The HUD needs to distinguish that from "sim silent".
        import time as _time

        _, mc = self._mc([], data={"highres_imu_mono": _time.monotonic()})
        s = mc.status()
        self.assertFalse(s["have_telemetry"])
        self.assertTrue(s["pose_blocked"])

    def test_status_pose_blocked_false_with_odometry(self):
        _, mc = self._mc([])
        self.assertFalse(mc.status()["pose_blocked"])

    def test_command_rate_within_sim_budget(self):
        # Sim spec §4.4 (fly2's VADR-TS-003): MAVLink command rate must stay
        # <= 100 Hz. Streaming faster made the sim ignore the whole setpoint
        # stream — armed drone, healthy HUD, zero motion.
        self.assertLessEqual(CONTROL_HZ, 100)

    def test_setpoint_slew_matches_control_rate(self):
        # SPACE/X slew the altitude target by CLIMB_RATE * CONTROL_DT_S per
        # tick; the climb rate is only honest if DT matches the actual loop.
        self.assertAlmostEqual(CONTROL_DT_S, 1.0 / CONTROL_HZ, places=9)

    def test_land_key_commands_descent(self):
        # L starts auto-land: the drone self-levels and commands < hover thrust.
        ctl, mc = self._mc(["l"])
        mc.tick()
        self.assertTrue(mc._landing)
        self.assertFalse(mc._landed)
        self.assertLess(ctl.last["thrust"], HOVER_T)

    def test_landing_touchdown_disarms(self):
        # After a real descent (vz > 0 in NED) followed by a sustained near-zero
        # vz (ground stops us), auto-land disarms and latches _landed.
        ctl, mc = self._mc(["l"])
        mc.tick()  # L pressed -> landing starts
        self.assertTrue(mc._landing)
        mc.data["odometry"]["vz"] = 1.0  # clearly descending
        mc.tick()
        mc.data["odometry"]["vz"] = 0.0  # ground stops the descent
        for _ in range(LAND_SETTLE_TICKS + 1):
            mc.tick()
        self.assertTrue(mc._landed)
        self.assertFalse(mc._landing)
        self.assertGreaterEqual(ctl.disarm_calls, 1)

    def test_landed_suppresses_rearm(self):
        # A deliberately-landed (disarmed) drone must not be re-armed by tick().
        ctl, mc = self._mc([], data=dict(level_data(), armed=False))
        mc._landed = True
        mc._next_arm_t = 0.0  # retry window elapsed
        mc.tick()
        self.assertEqual(ctl.arm_calls, 0)
        self.assertEqual(ctl.last["thrust"], 0.0)  # motors off on the ground

    def test_land_toggle_cancels(self):
        # A second L press cancels landing before touchdown (needs a fresh edge).
        held = {"l": False}
        ctl = FakeController()
        mc = ManualControl(
            ctl, dict(level_data(), armed=False), is_pressed=lambda k: held.get(k, False)
        )
        held["l"] = True
        mc.tick()  # press -> start landing
        self.assertTrue(mc._landing)
        held["l"] = False
        mc.tick()  # release
        held["l"] = True
        mc.tick()  # press again -> cancel
        self.assertFalse(mc._landing)
        self.assertFalse(mc._landed)


def _roll_quat(roll):
    """Quaternion (w, x, y, z) for a pure roll rotation."""
    return (math.cos(roll / 2.0), math.sin(roll / 2.0), 0.0, 0.0)


if __name__ == "__main__":
    unittest.main()
