"""Episodic reset harness for real-sim RL + a feasibility benchmark.

Real-sim RL lives or dies on being able to reset a fresh episode thousands of
times UNATTENDED. The VQ2 sim has no programmatic race-restart, but
`SimInterface.reset_sim()` (MAVLink cmd 31000) teleports the vehicle back to
spawn. After a teleport the IMU-integrated attitude is stale, so GPEstimation
must be re-seeded (`GPEstimation.reset()`). This module wraps that, and provides
`make rl2-reset-bench` to measure reset latency + re-align reliability BEFORE we
invest in the training loop. If resets are slow/flaky here, real-sim RL is not
viable and we fall back to the hybrid (headless bulk + real-sim fine-tune).

Also provides a vision-based `GatePassTracker` (the env counts gate passes from
vision + the sim's active_gate_index, since the race UI isn't scriptable).
"""

from __future__ import annotations

import argparse
import math
import time

import numpy as np

from simulator.gp_estimation import GPEstimation
from rl import spec
from rl.sim_interface import SimInterface


class RaceGo:
    """Faithful copy of GPPilot's WAIT_FOR_START -> GO gate (simulator/gp_pilot.py).

    `data["race_started"]` flips at countdown START -- flying then tips the drone
    over. Real GO waits for a FRESH countdown that has ELAPSED, with the physics
    clock live:
      * anchor sim_ms on the first race_status; re-anchor on a sim clock reset
        (Restart Race), matching the pilot.
      * race_fresh = start_ms > 0 and start_ms >= anchor  (not a stale finished race)
      * countdown_done = race_fresh and sim_ms >= start_ms and finish_ns < 0
      * physics_live = HIGHRES_IMU timestamp still advancing (frozen clock = idle
        sim; setpoints do nothing and release mid-command tips it over at launch).
    Prints the same [WAIT] lines as `make control-flight`. Call each tick."""

    CLOCK_RESET_SLACK_MS = 500
    IMU_FROZEN_S = 2.0
    DEBUG_EVERY_N = 45

    def __init__(self, debug: bool = True):
        self._anchor = None
        self._imu_ts_seen = None
        self._imu_ts_wall = 0.0
        self._debug = debug
        self._tick = 0

    def _physics_live(self, data: dict) -> bool:
        imu = data.get("imu")
        now = time.time()
        if imu is not None:
            ts = imu.get("time_us") or imu.get("time_usec")
            if ts != self._imu_ts_seen:
                self._imu_ts_seen = ts
                self._imu_ts_wall = now
        return self._imu_ts_wall > 0.0 and now - self._imu_ts_wall <= self.IMU_FROZEN_S

    def __call__(self, data: dict) -> bool:
        self._tick += 1
        physics_live = self._physics_live(data)
        race = data.get("race_status")
        if not race:
            if self._debug and self._tick % self.DEBUG_EVERY_N == 0:
                print("[WAIT] No race_status yet -- holding...", flush=True)
            return False
        sim_ms = int(race.get("sim_boot_time_ms", 0) or 0)
        start_ms = int(race.get("race_start_boot_time_ms", -1) or -1)
        if self._anchor is None:
            self._anchor = sim_ms
            print(f"[WAIT] Anchor set: sim_ms={sim_ms}", flush=True)
        if sim_ms < self._anchor - self.CLOCK_RESET_SLACK_MS:
            print(f"[WAIT] Sim clock reset (sim_ms={sim_ms} < anchor={self._anchor}) "
                  "-- re-anchoring for the new race.", flush=True)
            self._anchor = None                      # re-anchor next tick (prints again)
            return False
        finish_ns = int(race.get("race_finish_time_ns", -1) or -1)
        race_fresh = start_ms > 0 and start_ms >= self._anchor
        countdown_done = race_fresh and sim_ms >= start_ms and finish_ns < 0
        go = countdown_done and physics_live
        if self._debug and self._tick % self.DEBUG_EVERY_N == 0:
            print(f"[WAIT] sim_ms={sim_ms} race_start={start_ms} finish_ns={finish_ns} "
                  f"fresh={race_fresh} go={go}", flush=True)
        return go


def _est_ready(sim: SimInterface, est: GPEstimation) -> bool:
    """Fresh IMU + camera flowing and GPEstimation producing a finite attitude."""
    if sim.data.get("imu") is None or not sim.data.get("frame"):
        return False
    q = np.asarray(est.snapshot()["quat"], dtype=np.float64)
    return bool(np.all(np.isfinite(q)) and abs(np.linalg.norm(q) - 1.0) < 0.2)


def wait_ready(sim: SimInterface, est: GPEstimation, timeout_s: float = 8.0) -> bool:
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        if _est_ready(sim, est):
            return True
        time.sleep(0.02)
    return False


def reset_episode(sim: SimInterface, est: GPEstimation, settle_s: float = 1.0,
                  timeout_s: float = 8.0) -> bool:
    """Teleport to spawn, re-seed the estimator, wait until telemetry + attitude
    are healthy, and re-arm. Returns True if the episode is ready to fly."""
    sim.reset_sim()                 # cmd 31000 teleport
    time.sleep(0.3)                 # let the teleport land + RX catch up
    est.reset()                     # re-seed AHRS (teleport invalidates integration)
    ok = wait_ready(sim, est, timeout_s=timeout_s)
    if settle_s > 0:
        time.sleep(settle_s)        # let the drone settle at spawn before flying
    sim.arm()
    return ok


class GatePassTracker:
    """Count gate passes for the reward. Prefers the sim's `active_gate_index`
    (authoritative if the race is live); otherwise detects a pass from VISION:
    the range to the nearest gate dropped close (< NEAR_M) and then JUMPED UP
    (the NEXT gate appeared far) -- which is what a real pass looks like on a
    continuous course. Robust to the acquired/lost flicker (a same-gate
    re-acquire has ~the same range, so no jump -> no false pass)."""

    NEAR_M = 2.0   # "at the gate" within this range
    JUMP_M = 3.0   # range growing this far past its minimum = switched to next gate

    def __init__(self):
        self.n_passed = 0
        self._min_r = float("inf")
        self._last_index = None

    def reset(self, sim: SimInterface) -> None:
        self.n_passed = 0
        self._min_r = float("inf")
        self._last_index = int(sim.data.get("active_gate_index", 0) or 0)

    def update(self, sim: SimInterface, pose_est: dict | None) -> bool:
        # 1) authoritative: race index advanced (only if the race is live).
        idx = sim.data.get("active_gate_index")
        if idx is not None and self._last_index is not None and int(idx) > self._last_index:
            self._last_index = int(idx)
            self.n_passed += 1
            self._min_r = float("inf")
            return True
        if idx is not None:
            self._last_index = int(idx)
        # 2) vision: got close, then the range jumped up -> passed to the next gate.
        if pose_est is not None:
            r = math.sqrt(pose_est["body_x_m"] ** 2 + pose_est["body_y_m"] ** 2
                          + pose_est["body_z_m"] ** 2)
            if r < self._min_r:
                self._min_r = r
            if self._min_r < self.NEAR_M and r > self._min_r + self.JUMP_M:
                self.n_passed += 1
                self._min_r = r          # re-baseline toward the new gate
                return True
        return False


def benchmark(cycles: int = 10, hover_s: float = 1.5) -> None:
    """Run N automated reset -> brief hover -> reset cycles; report latency +
    re-align reliability. This is the go/no-go for real-sim RL."""
    sim = SimInterface()
    est = GPEstimation(sim.data)
    est.start()
    print(f"[rl2-reset-bench] connected; running {cycles} reset cycles...", flush=True)
    lat = []
    ok_count = 0
    for i in range(cycles):
        t0 = time.time()
        ok = reset_episode(sim, est, settle_s=0.5)
        dt = time.time() - t0
        lat.append(dt)
        ok_count += int(ok)
        snap = est.snapshot()
        att = snap["att_deg"]
        print(f"  cycle {i+1:2d}: reset {dt:4.1f}s  ready={ok}  "
              f"att=({att[0]:+.0f}r,{att[1]:+.0f}p,{att[2]:+.0f}y)", flush=True)
        # brief neutral hover so the next reset starts from motion (realistic).
        t1 = time.time()
        while time.time() - t1 < hover_s:
            sim.send_attitude_rates(0.0, 0.0, 0.0, spec.HOVER_THRUST)
            time.sleep(1 / 50)
    lat = np.array(lat)
    print("\n[rl2-reset-bench] RESULT:", flush=True)
    print(f"  reset latency (s): mean={lat.mean():.2f} min={lat.min():.2f} max={lat.max():.2f}")
    print(f"  re-align success:  {ok_count}/{cycles}")
    est_ep = lat.mean() + 6.0  # ~reset + a short episode
    print(f"  => ~{est_ep:.0f}s per (reset+episode). For 200k steps @50Hz ({200000/50/60:.0f} min flight)")
    print("     plus resets, budget accordingly. FEASIBLE if latency a few s, success ~100%.")
    sim.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cycles", type=int, default=10)
    ap.add_argument("--hover", type=float, default=1.5)
    args = ap.parse_args()
    benchmark(args.cycles, args.hover)
