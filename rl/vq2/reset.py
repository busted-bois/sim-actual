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
from simulator.gp_vision import _yolo_pose_estimate
from rl import spec
from rl.sim_interface import SimInterface


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
    (ground-truth progress counter, if the race is live); falls back to a
    vision near-then-lost heuristic when the index isn't advancing."""

    def __init__(self, near_m: float = 2.5):
        self.near_m = near_m
        self.n_passed = 0
        self._was_near = False
        self._last_index = None

    def reset(self, sim: SimInterface) -> None:
        self.n_passed = 0
        self._was_near = False
        self._last_index = int(sim.data.get("active_gate_index", 0) or 0)

    def update(self, sim: SimInterface, pose_est: dict | None) -> bool:
        # 1) authoritative: race index advanced.
        idx = sim.data.get("active_gate_index")
        if idx is not None and self._last_index is not None and int(idx) > self._last_index:
            self._last_index = int(idx)
            self.n_passed += 1
            self._was_near = False
            return True
        if idx is not None:
            self._last_index = int(idx)
        # 2) fallback: was within near_m of the gate, then lost it (crossed plane).
        if pose_est is not None:
            r = math.sqrt(pose_est["body_x_m"] ** 2 + pose_est["body_y_m"] ** 2
                          + pose_est["body_z_m"] ** 2)
            if r < self.near_m:
                self._was_near = True
        elif self._was_near:
            self._was_near = False
            self.n_passed += 1
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
    print(f"     plus resets, budget accordingly. FEASIBLE if latency is a few s and success ~100%.")
    sim.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cycles", type=int, default=10)
    ap.add_argument("--hover", type=float, default=1.5)
    args = ap.parse_args()
    benchmark(args.cycles, args.hover)
