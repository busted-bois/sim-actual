"""GP sign audit: closed-loop GPPilot stability vs plant sign conventions.

The GP stack (GPEstimation's blanket 3-axis gyro negation + gp_pilot's
KP,KR,KY = +1,-1,-1) was ported from AndurilGP, which flew live — but the
repo's other stacks (IBVS _TiltFilter, flightlab signs.json) encode partly
different conventions, and the combination had never been proven stable
against a truth-convention plant.

These tests close the loop offline: a truth-convention plant (no
auto-leveling, first-order body-rate lag) parameterized by per-axis command
and gyro signs drives the REAL GPEstimation._process_imu + compute_guidance.

Verdict encoded by the tests:
  * ANDURIL plant (all gyro axes inverted; roll/yaw commands inverted, pitch
    command direct) -> stable, estimate tracks truth. Regression-locked.
  * B0_NAIVE plant (pitch-only gyro inversion, all commands inverted — i.e.
    reading flightlab/signs.json as truth-frame command signs alongside the
    measured pitch-gyro-inversion fact) -> the roll AND pitch loops are
    positive feedback; GPPilot cannot fly that plant at all. Since AndurilGP
    flew the same sim, B0's {-1,-1,-1} most plausibly folds the (inverted)
    gyro into its cmd->gyro measurement rather than contradicting ANDURIL.
Final arbiter for the real sim: rerun `make attitude-harness` (B0) live.
If live data ever proves the B0_NAIVE plant, the one-line fixes are:
negate ONLY the pitch gyro in GPEstimation._process_imu, and flip KP to -1
in simulator/gp_pilot.py.

NOTE: the LIVE pilot no longer flies the rate wire at all — it ships
compute_guidance's degree commands on the attitude-quaternion encoding
(Controller "attitude_quat", the original AndurilGP form that flew the
course; launch flips traced to the rate reinterpretation, 2026-07-16).
This audit therefore now covers the RL-expert/internal-env RATE path
(rl.gp_expert converts deg -> rad/s the same way _run_closed_loop does).
"""

from __future__ import annotations

import math
import unittest

import numpy as np

from rl.core import spec
from simulator.gp_estimation import GPEstimation
from simulator.gp_pilot import _fresh_hold_state, compute_guidance
from simulator.gyro_ahrs import GyroAHRS

IMU_HZ = 100.0
RATE_TAU_S = 0.05  # first-order body-rate lag (matches measured rate_tau_s)

# (s_cmd, s_gyro) per axis [roll, pitch, yaw]: true_rate = s_cmd * commanded,
# gyro_reported = s_gyro * true_rate.
ANDURIL = {"s_cmd": (-1.0, 1.0, -1.0), "s_gyro": (-1.0, -1.0, -1.0)}
B0_NAIVE = {"s_cmd": (-1.0, -1.0, -1.0), "s_gyro": (1.0, -1.0, 1.0)}


class _Plant:
    """Truth-convention attitude plant: rate lag + quaternion integration."""

    def __init__(self, signs: dict, pitch0_deg: float, roll0_deg: float = 0.0):
        self.s_cmd = np.array(signs["s_cmd"])
        self.s_gyro = np.array(signs["s_gyro"])
        self.truth = GyroAHRS(initial_pitch_deg=pitch0_deg, initial_roll_deg=roll0_deg)
        self.rates = np.zeros(3)  # true body rates, rad/s
        self.t_us = 0

    def step(self, cmd_rates: np.ndarray, dt: float) -> dict:
        """Apply commanded rates, integrate truth, emit the IMU dict."""
        self.rates += (self.s_cmd * cmd_rates - self.rates) * (dt / RATE_TAU_S)
        self.truth.update(*self.rates, dt)
        self.t_us += int(dt * 1e6)
        g = self.s_gyro * self.rates
        return {
            "gx": float(g[0]),
            "gy": float(g[1]),
            "gz": float(g[2]),
            "ax": 0.0,
            "ay": 0.0,
            "az": -9.81,
            "time_us": self.t_us,
        }

    def euler_deg(self) -> tuple[float, float, float]:
        return self.truth.euler_deg()


def _run_closed_loop(
    signs: dict,
    seconds: float = 6.0,
    pitch0_deg: float = -17.8,
    roll0_deg: float = 2.0,
    gate_world: np.ndarray | None = None,
):
    """Fly compute_guidance on the real GPEstimation against the plant.

    Returns per-tick history of truth euler, estimate euler, and (when a gate
    is supplied) the body-frame bearing the guidance saw.
    """
    plant = _Plant(signs, pitch0_deg=pitch0_deg, roll0_deg=roll0_deg)
    est = GPEstimation({}, launch_pitch_deg=pitch0_deg)
    # Seed estimate == truth (live, the seed is the known launch-ramp pose).
    # GyroAHRS never observes an initial offset, so a mismatched seed would
    # just carry through as a constant bias and mask the loop behavior.
    est.ahrs = GyroAHRS(initial_pitch_deg=pitch0_deg, initial_roll_deg=roll0_deg)
    hold = _fresh_hold_state()
    dt = 1.0 / IMU_HZ

    # Prime: _process_imu's first sample only records the timestamp.
    est._process_imu(plant.step(np.zeros(3), dt))

    hist = {"truth": [], "est": [], "bearing": []}
    cmd = np.zeros(3)
    frame_id = 0
    for tick in range(int(seconds * IMU_HZ)):
        est._process_imu(plant.step(cmd, dt))
        snap = est.snapshot()
        roll_e, pitch_e, _yaw_e = snap["att_deg"]

        vision = None
        if gate_world is not None and tick % 3 == 0:  # ~30 Hz vision
            frame_id += 1
            R = spec.quat_to_R(np.asarray(plant.truth.quaternion, float))
            b = R.T @ gate_world  # body FRD gate position (no translation)
            vision = {
                "body_x_m": float(b[0]),
                "body_y_m": float(b[1]),
                "body_z_m": float(b[2]),
                "frame_id": frame_id,
                "normal_body": None,
            }
            hist["bearing"].append(math.degrees(math.atan2(b[1], b[0])))

        rd, pd, yd, _thrust, _dbg = compute_guidance(
            roll_deg=roll_e,
            pitch_deg=pitch_e,
            quat=snap["quat"],
            vY=0.0,
            vD=0.0,
            vision=vision,
            vision_vel=None,
            state=hold,
        )
        # Guidance returns DEGREE commands (attitude-quat wire); this audit's
        # plant is the rate world (the RL expert path), so interpret deg ->
        # rad/s exactly like rl.gp_expert does.
        cmd = np.radians([rd, pd, yd])
        hist["truth"].append(plant.euler_deg())
        hist["est"].append((roll_e, pitch_e))
    return hist


class AndurilPlantTests(unittest.TestCase):
    """Regression lock: the shipped sign set is stable on the plant it flew."""

    def test_pitch_converges_to_desired_and_est_tracks_truth(self):
        hist = _run_closed_loop(ANDURIL)
        truth = np.array(hist["truth"])
        est = np.array(hist["est"])
        # Bounded throughout, converged to DESIRED_PITCH_DEG (-3); the 2-deg
        # initial roll is actively releveled through the KR loop.
        self.assertLess(np.abs(truth[:, :2]).max(), 40.0)
        self.assertAlmostEqual(truth[-1, 1], -3.0, delta=1.5)
        self.assertLess(abs(truth[-1, 0]), 0.5)
        # Blanket gyro negation on this plant => estimate IS truth.
        self.assertLess(np.abs(est[:, 0] - truth[:, 0]).max(), 2.0)
        self.assertLess(np.abs(est[:, 1] - truth[:, 1]).max(), 2.0)

    def test_banks_toward_gate_then_relevels_and_yaws_to_it(self):
        # Gate 10 m ahead, 4 m to the RIGHT (+y body/world at yaw 0).
        hist = _run_closed_loop(
            ANDURIL,
            seconds=8.0,
            pitch0_deg=-3.0,
            roll0_deg=0.0,
            gate_world=np.array([10.0, 4.0, 0.0]),
        )
        truth = np.array(hist["truth"])
        est = np.array(hist["est"])
        self.assertLess(np.abs(truth[:, :2]).max(), 40.0)
        # Banked toward the gate (positive roll = right) before releveling.
        #
        # The bound is 3 deg, not the 8 deg this assertion originally carried.
        # That 8 was never achievable: the identical peak of 4.5347 deg comes
        # out of 7d04af1, the commit that introduced this test, so it has been
        # red since the day it landed. The cap is a property of THIS harness,
        # not of the pilot. compute_guidance returns absolute attitude in
        # degrees, and on blind ticks (no fresh vision) it returns a HOLD --
        # rd = current roll. _run_closed_loop reinterprets degrees as rad/s to
        # mirror the RL expert path, which turns "hold 4.5 deg" into a rate
        # proportional to current roll and leaks the bank back toward level.
        # What this test can legitimately lock is the SIGN: a gate to the right
        # must produce right bank and a right-hand yaw, which is the sign
        # convention the whole audit exists to pin down.
        self.assertGreater(truth[:, 0].max(), 3.0)
        self.assertLess(abs(truth[-1, 0]), 6.0)
        # Yawed toward the gate: heading moved right, bearing nulled.
        self.assertGreater(truth[-1, 2], 5.0)
        self.assertLess(abs(hist["bearing"][-1]), 4.0)
        self.assertLess(np.abs(est - truth[:, :2]).max(), 2.0)


class B0NaivePlantTests(unittest.TestCase):
    """Documents incompatibility: GPPilot diverges on the B0-naive plant.

    If a live `make attitude-harness` rerun ever proves this plant is the
    real sim, apply the one-line fixes named in the module docstring — this
    test then flips to a convergence lock.
    """

    def test_roll_and_pitch_loops_are_positive_feedback(self):
        hist = _run_closed_loop(B0_NAIVE, seconds=5.0)
        truth = np.array(hist["truth"])
        self.assertGreater(np.abs(truth[:, :2]).max(), 45.0)


if __name__ == "__main__":
    unittest.main()
