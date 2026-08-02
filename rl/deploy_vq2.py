"""Fly the VQ2 policy on the LIVE simulator — vision + IMU only.

This is the runner rl/deploy.py could never be: it needs no gate map, no
odometry, no world pose. Everything it reads was confirmed present in a live
Qualification session by simulator/telemetry_probe.py on 2026-08-01:

    HIGHRES_IMU          114.0 Hz   (gyro clean; mag/baro NaN)
    ENCAPSULATED_DATA      4.4 Hz   race_status -> active_gate_index
    camera UDP 5600         30 Hz   -> YOLO-pose -> PnP
    ODOMETRY / ATTITUDE / LOCAL_POSITION_NED   absent (requested, withheld)
    track_info                0     no gate poses

Two modes:

  --observe   Read-only. Runs the whole perception -> observation chain against
              the live sim and logs the vector, but NEVER arms and NEVER sends a
              setpoint. Safe during someone else's flight. Run this FIRST: it
              proves the pipeline on real data with nothing at risk.

  --fly       Arms and closes the loop.

Calibration warning: this repo has never run `make attitude-harness`, so
flightlab/calibration.json and signs.json do not exist and three mutually
inconsistent command-sign conventions are in the tree — one of which is
recorded as having flown the drone upside down. The signs and rate gain below
are therefore DEFAULTS, not measurements, and every one is overridable from the
environment so they can be tuned live without a code change:

    RL_SIGN_ROLL RL_SIGN_PITCH RL_SIGN_YAW RL_RATE_GAIN
    RL_GYRO_SIGN RL_RATE_CLIP RL_THRUST_MIN RL_THRUST_MAX RL_HZ

    uv run -m rl.deploy_vq2 --observe
    uv run -m rl.deploy_vq2 --fly
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import time

import numpy as np

from rl.core import vq2_observation as vo

# Command-sign defaults follow rl/experts/fly2_course.py, the rate-wire pilot.
SIGN_ROLL = float(os.environ.get("RL_SIGN_ROLL", -1.0))
SIGN_PITCH = float(os.environ.get("RL_SIGN_PITCH", +1.0))
SIGN_YAW = float(os.environ.get("RL_SIGN_YAW", -1.0))
# The live plant amplifies rate commands ~2.7x (memory, live-confirmed, never
# re-measured on this machine).
RATE_GAIN = float(os.environ.get("RL_RATE_GAIN", 2.7))
# simulator/gp_estimation.py negates all three gyro axes; that pilot is the one
# that actually flies, so match it by default.
GYRO_SIGN = float(os.environ.get("RL_GYRO_SIGN", -1.0))
RATE_CLIP = float(os.environ.get("RL_RATE_CLIP", 1.2))  # rad/s at the wire
THRUST_MIN = float(os.environ.get("RL_THRUST_MIN", 0.05))
THRUST_MAX = float(os.environ.get("RL_THRUST_MAX", 0.85))
CONTROL_HZ = float(os.environ.get("RL_HZ", 50.0))  # match training DECISION_HZ

N_GATES = int(os.environ.get("RL_N_GATES", 17))
STALL_TIMEOUT_S = float(os.environ.get("RL_STALL_S", 20.0))
LOG_DIR = os.path.join("rl", "data")


def _gravity_from_accel(imu) -> np.ndarray | None:
    """Unit gravity direction in body frame from a resting accel sample."""
    if not imu:
        return None
    a = np.array(
        [imu.get("xacc", 0.0), imu.get("yacc", 0.0), imu.get("zacc", 0.0)], float
    )
    n = float(np.linalg.norm(a))
    if not np.isfinite(n) or n < 1e-3:
        return None
    # At rest the accelerometer reads -g (measured on the pad: |a| = 9.8097 at
    # 17.80 deg nose-down), so gravity points opposite the reading.
    return -a / n


class VQ2Pilot:
    """Perception -> VQ2 observation -> policy -> attitude-rate command."""

    def __init__(self, data, controller, policy=None, n_gates=N_GATES, log=None):
        from simulator.gyro_ahrs import GyroAHRS

        self.data = data
        self.controller = controller
        self.policy = policy
        self.tracker = vo.GateFeatureTracker(n_gates=n_gates)
        # Seed at the measured start attitude; refined from the first accel.
        self.ahrs = GyroAHRS(initial_pitch_deg=-17.8)
        self._seeded = False
        self._last_imu_t = None
        self._last_action = np.zeros(4)
        self._state = None
        self._first = True
        self._t0 = time.monotonic()
        self._last_gate = 0
        self._last_gate_t = self._t0
        self.gates_passed = 0
        self._log = log
        self._ticks = 0
        self._det_ticks = 0

    # -- sensing ---------------------------------------------------------
    def _gyro(self) -> np.ndarray:
        imu = self.data.get("imu") or {}
        g = np.array(
            [imu.get("xgyro", 0.0), imu.get("ygyro", 0.0), imu.get("zgyro", 0.0)],
            dtype=float,
        )
        g = np.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0)
        return GYRO_SIGN * g

    def _gravity(self) -> np.ndarray:
        """Gravity in body from the gyro-integrated AHRS.

        Accel is only trustworthy at rest (it is corrupted by thrust in
        flight), so it seeds the attitude once and the gyro carries it after.
        """
        imu = self.data.get("imu") or {}
        # HIGHRES_IMU.time_usec is FROZEN on this sim (measured: 1141 samples,
        # 0 us span), so dt must come from arrival time, never the stamp.
        now = time.monotonic()
        dt = 0.0 if self._last_imu_t is None else max(now - self._last_imu_t, 0.0)
        self._last_imu_t = now

        if not self._seeded:
            g = _gravity_from_accel(imu)
            if g is not None and abs(float(np.linalg.norm(g)) - 1.0) < 0.2:
                pitch = math.degrees(math.atan2(g[0], g[2]))
                roll = math.degrees(math.atan2(-g[1], g[2]))
                self.ahrs = type(self.ahrs)(
                    initial_pitch_deg=pitch, initial_roll_deg=roll
                )
                self._seeded = True
                print(
                    f"[vq2] AHRS seeded from accel: roll={roll:+.1f} "
                    f"pitch={pitch:+.1f} deg",
                    flush=True,
                )

        gx, gy, gz = self._gyro()
        if 0.0 < dt < 0.5:
            self.ahrs.update(gx, gy, gz, dt)
        q = self.ahrs.q
        w, x, y, z = q
        # gravity (world +z down) rotated into body = third ROW of R_wb.
        return np.array(
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]
        )

    def _vision(self):
        from simulator.gp_vision import best_pose_gate

        g = best_pose_gate(self.data)
        if not g:
            return None
        p = g.get("pose") or {}
        if "gate_pos_body" not in p:
            return None
        est = dict(p)
        est["conf"] = float(g.get("conf", 0.0))
        return est

    def _gate_index(self) -> int:
        """Authoritative progress from race_status — the only oracle VQ2 gives."""
        return int(self.data.get("active_gate_index", 0) or 0)

    # -- control ---------------------------------------------------------
    def observation(self) -> np.ndarray:
        est = self._vision()
        if est is not None:
            self._det_ticks += 1
        idx = self._gate_index()
        if idx > self._last_gate:
            self._last_gate = idx
            self.gates_passed = idx
            self._last_gate_t = time.monotonic()
            print(
                f"[vq2] gate {idx} passed at {time.monotonic() - self._t0:.1f}s",
                flush=True,
            )
        return self.tracker.update(
            time.monotonic(),
            est,
            self._gyro(),
            self._gravity(),
            self._last_action,
            idx,
        )

    def act(self, obs) -> np.ndarray:
        if self.policy is None:
            return np.zeros(4)
        a, self._state = self.policy.predict(
            obs[None],
            state=self._state,
            episode_start=np.array([self._first]),
            deterministic=True,
        )
        self._first = False
        return np.asarray(a[0], dtype=float)

    def tick(self):
        self._ticks += 1
        obs = self.observation()
        action = np.clip(self.act(obs), -1.0, 1.0)
        self._last_action = action

        from rl.core import spec

        roll, pitch, yaw, thrust = spec.scale_action(action)
        # Undo the plant's rate amplification, then apply measured-sign remap.
        roll = float(np.clip(SIGN_ROLL * roll / RATE_GAIN, -RATE_CLIP, RATE_CLIP))
        pitch = float(np.clip(SIGN_PITCH * pitch / RATE_GAIN, -RATE_CLIP, RATE_CLIP))
        yaw = float(np.clip(SIGN_YAW * yaw / RATE_GAIN, -RATE_CLIP, RATE_CLIP))
        thrust = float(np.clip(thrust, THRUST_MIN, THRUST_MAX))

        if self.controller is not None:
            self.controller.set_attitude_rates(roll, pitch, yaw, thrust)
        if self._log:
            self._log.writerow(
                [
                    f"{time.monotonic() - self._t0:.3f}",
                    self.gates_passed,
                    f"{roll:.4f}",
                    f"{pitch:.4f}",
                    f"{yaw:.4f}",
                    f"{thrust:.4f}",
                ]
                + [f"{v:.4f}" for v in obs]
            )

    def stalled(self) -> bool:
        return (time.monotonic() - self._last_gate_t) > STALL_TIMEOUT_S

    def report(self):
        det = 100.0 * self._det_ticks / max(self._ticks, 1)
        print(
            f"[vq2] {self._ticks} ticks, gates={self.gates_passed}, "
            f"vision on {det:.1f}% of ticks, "
            f"{time.monotonic() - self._t0:.1f}s elapsed",
            flush=True,
        )


def load_policy(path: str):
    from sb3_contrib import RecurrentPPO

    if not os.path.exists(path):
        raise SystemExit(
            f"[vq2] no policy at {path} — run `make train-vq2` first "
            "(or use --observe, which needs no policy)"
        )
    print(f"[vq2] loading {path}", flush=True)
    return RecurrentPPO.load(path, device="cpu")


def main():
    ap = argparse.ArgumentParser()
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--observe",
        action="store_true",
        help="read-only: log the observation, never arm, never command",
    )
    mode.add_argument("--fly", action="store_true", help="arm and close the loop")
    ap.add_argument("--policy", default=os.path.join("rl", "data", "vq2", "ppo_vq2"))
    ap.add_argument("--seconds", type=float, default=0.0, help="0 = until Ctrl+C")
    ap.add_argument("--n-gates", type=int, default=N_GATES)
    args = ap.parse_args()

    os.environ.setdefault("AUTO_PILOT", "none")
    from simulator.setup import setup_components

    shared: dict = {}
    boot_ms = int(time.time() * 1000)
    comps = setup_components(shared, boot_ms, "127.0.0.1", 14550)
    controller = comps["controller"]

    policy = None if args.observe else load_policy(args.policy)

    os.makedirs(LOG_DIR, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(
        LOG_DIR, f"vq2_{'obs' if args.observe else 'fly'}_{stamp}.csv"
    )
    fh = open(log_path, "w", newline="")
    writer = csv.writer(fh)
    writer.writerow(
        ["t", "gate", "roll", "pitch", "yaw", "thrust"]
        + [
            f"{k}{i}"
            for k, sl in vo.OBS_LAYOUT.items()
            for i in range(sl.stop - sl.start)
        ]
    )

    pilot = VQ2Pilot(
        shared,
        None if args.observe else controller,
        policy=policy,
        n_gates=args.n_gates,
        log=writer,
    )
    controller.pilot = pilot
    controller.control_hz = CONTROL_HZ

    if args.observe:
        print("[vq2] OBSERVE mode — not arming, not commanding.", flush=True)
    else:
        print(
            f"[vq2] FLY mode — signs=({SIGN_ROLL:+.0f},{SIGN_PITCH:+.0f},"
            f"{SIGN_YAW:+.0f}) rate_gain={RATE_GAIN} clip=±{RATE_CLIP} rad/s "
            f"thrust=[{THRUST_MIN},{THRUST_MAX}] @ {CONTROL_HZ:.0f} Hz",
            flush=True,
        )
        controller.arm()

    t0 = time.monotonic()
    period = 1.0 / CONTROL_HZ
    try:
        while True:
            loop = time.monotonic()
            if args.observe:
                pilot.tick()  # builds + logs the observation, commands nothing
            else:
                controller.update()  # calls pilot.tick(), then sends the setpoint
            if args.seconds and (time.monotonic() - t0) > args.seconds:
                break
            if args.fly and pilot.stalled():
                print(
                    f"[vq2] no gate progress for {STALL_TIMEOUT_S:.0f}s — stopping",
                    flush=True,
                )
                break
            slack = period - (time.monotonic() - loop)
            if slack > 0:
                time.sleep(slack)
    except KeyboardInterrupt:
        print("\n[vq2] interrupted", flush=True)
    finally:
        pilot.report()
        fh.close()
        print(f"[vq2] log -> {log_path}", flush=True)
        for k in ("ts_loop", "mavlink_rx", "vision_rx"):
            c = comps.get(k)
            if c is not None and hasattr(c, "get_thread_for_join"):
                c.get_thread_for_join().join(timeout=2.0)


if __name__ == "__main__":
    main()
