"""Offline units for the GyroAHRS complementary attitude filter (no sim).

The complementary path fuses gyro integration (accurate short-term, drifts)
with the accelerometer gravity tilt (stable long-term, corrupted by thrust),
gated to samples where |accel| ~= g. Accel signs follow this sim's
specific-force convention: a static, level body reads az ~= -g.
"""

import math
import unittest

from simulator.gyro_ahrs import G, GyroAHRS


def _level_accel():
    """Body accel of a static, level drone in this sim (az ~= -g)."""
    return 0.0, 0.0, -G


class ComplementaryFilterTests(unittest.TestCase):
    def test_alpha_one_matches_pure_gyro(self):
        # alpha=1 => accel ignored => identical to the gyro-only update().
        a = GyroAHRS(initial_pitch_deg=-17.8)
        b = GyroAHRS(initial_pitch_deg=-17.8)
        gx, gy, gz = 0.2, -0.1, 0.05
        for _ in range(50):
            r_pure = a.update(gx, gy, gz, 0.01)
            ax, ay, az = _level_accel()
            r_comp = b.update_with_accel(gx, gy, gz, ax, ay, az, 0.01, alpha=1.0)
        self.assertAlmostEqual(r_pure[0], r_comp[0], places=6)
        self.assertAlmostEqual(r_pure[1], r_comp[1], places=6)
        self.assertAlmostEqual(r_pure[2], r_comp[2], places=6)

    def test_level_accel_gives_zero_tilt(self):
        # Wrong initial tilt + zero gyro: accel must pull roll/pitch toward 0.
        # Correction is slew-limited (drift-rate), so give it plenty of time.
        ahrs = GyroAHRS(initial_pitch_deg=20.0, initial_roll_deg=15.0)
        ax, ay, az = _level_accel()
        for _ in range(8000):
            roll, pitch, _ = ahrs.update_with_accel(0.0, 0.0, 0.0, ax, ay, az, 0.01)
        self.assertAlmostEqual(roll, 0.0, places=1)
        self.assertAlmostEqual(pitch, 0.0, places=1)

    def test_static_tilt_converges_to_accel_angle(self):
        # Held at a real roll: gravity splits into body Y. roll = atan2(-ay,-az).
        roll_true = math.radians(20.0)
        ay = -G * math.sin(roll_true)
        az = -G * math.cos(roll_true)
        ax = 0.0
        ahrs = GyroAHRS(initial_pitch_deg=0.0, initial_roll_deg=0.0)
        for _ in range(8000):
            roll, pitch, _ = ahrs.update_with_accel(0.0, 0.0, 0.0, ax, ay, az, 0.01)
        self.assertAlmostEqual(roll, 20.0, places=0)
        self.assertAlmostEqual(pitch, 0.0, places=0)

    def test_pitch_sign_nose_up(self):
        # Nose-up pitch: gravity leaks into +ax (pitch = atan2(ax, hypot(ay,az))).
        pitch_true = math.radians(15.0)
        ax = G * math.sin(pitch_true)
        ay = 0.0
        az = -G * math.cos(pitch_true)
        ahrs = GyroAHRS()
        for _ in range(8000):
            _, pitch, _ = ahrs.update_with_accel(0.0, 0.0, 0.0, ax, ay, az, 0.01)
        self.assertAlmostEqual(pitch, 15.0, places=0)

    def test_banked_hover_thrust_cannot_flatten_estimate(self):
        # Crash-regression: a quad leaning toward a gate reads thrust ~= 1 g
        # along body -z, i.e. accel claims "level" while the drone is banked.
        # The slew limit must keep the estimate near the true bank over a
        # gate-approach timescale instead of dragging it to 0 (which made the
        # pilot over-bank and fly into the first gate).
        ahrs = GyroAHRS(initial_roll_deg=8.0)
        ax, ay, az = _level_accel()  # thrust-only reading, |a| = g
        for _ in range(300):  # 3 s at 100 Hz, zero rates (steady lean)
            roll, _, _ = ahrs.update_with_accel(0.0, 0.0, 0.0, ax, ay, az, 0.01)
        self.assertGreater(roll, 6.0)  # <= 0.5 deg/s leak => >= 6.5 deg left

    def test_rate_gate_blocks_correction_while_rotating(self):
        # While manoeuvring (large gyro rates) the accel tilt is meaningless;
        # the output must equal the pure-gyro path exactly.
        comp = GyroAHRS(initial_pitch_deg=-17.8)
        gyro = GyroAHRS(initial_pitch_deg=-17.8)
        gx, gy, gz = 0.3, -0.2, 0.1  # all above RATE_GATE_RAD_S
        ax, ay, az = _level_accel()  # |a| = g, would pass the accel gate
        for _ in range(200):
            r_comp = comp.update_with_accel(gx, gy, gz, ax, ay, az, 0.01)
            r_gyro = gyro.update(gx, gy, gz, 0.01)
        self.assertAlmostEqual(r_comp[0], r_gyro[0], places=6)
        self.assertAlmostEqual(r_comp[1], r_gyro[1], places=6)
        self.assertAlmostEqual(r_comp[2], r_gyro[2], places=6)

    def test_thrust_spike_gate_off_is_pure_gyro(self):
        # |accel| way above g (thrust/manoeuvre): correction must not apply,
        # so the estimate equals the gyro-only path exactly.
        comp = GyroAHRS(initial_pitch_deg=-17.8)
        gyro = GyroAHRS(initial_pitch_deg=-17.8)
        gx, gy, gz = 0.15, 0.1, -0.05
        ax, ay, az = 0.0, 0.0, -3.0 * G  # 3 g specific force
        for _ in range(100):
            r_comp = comp.update_with_accel(gx, gy, gz, ax, ay, az, 0.01)
            r_gyro = gyro.update(gx, gy, gz, 0.01)
        self.assertAlmostEqual(r_comp[0], r_gyro[0], places=6)
        self.assertAlmostEqual(r_comp[1], r_gyro[1], places=6)
        self.assertAlmostEqual(r_comp[2], r_gyro[2], places=6)

    def test_yaw_stays_gyro_only(self):
        # Accel is blind to heading: pure yaw rate integrates unaffected.
        ahrs = GyroAHRS()
        ax, ay, az = _level_accel()
        for _ in range(100):
            _, _, yaw = ahrs.update_with_accel(0.0, 0.0, 0.5, ax, ay, az, 0.01)
        self.assertGreater(yaw, 10.0)  # yaw accumulated from the 0.5 rad/s rate


if __name__ == "__main__":
    unittest.main()
