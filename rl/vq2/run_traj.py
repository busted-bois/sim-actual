"""Replay a saved trajectory's ACTIONS open-loop in the real simulator.

This is the milestone-10 verification: it takes a trajectory saved by
`rl2-log-demos` and re-sends its recorded action sequence through the EXACT same
path the RL policy/servo uses -- `controller.action_to_attitude` ->
`SimInterface.send_attitude_quat_deg` -- at the recorded timing. If the drone
flies the same course as the original demonstration, the save + action-channel
are faithful (and the policy trained on these actions will reproduce the flight).

No neural net, no pilot -- just the recorded commands.

    # start (or restart) the race in the sim, then:
    make rl2-run ARGS="demo_1737764812"      # a save name from rl/data/vq2/saves/
"""

from __future__ import annotations

import argparse
import os
import time

import numpy as np

from rl.vq2 import controller
from rl.vq2.reset import RaceGo
from rl.sim_interface import SimInterface

DATA = os.path.join("rl", "data", "vq2")
SAVES = os.path.join(DATA, "saves")


def _resolve(name: str) -> str:
    """Accept a bare name, a name.npz, or a full path."""
    for cand in (name, name + ".npz", os.path.join(SAVES, name),
                 os.path.join(SAVES, name + ".npz")):
        if os.path.exists(cand):
            return cand
    raise FileNotFoundError(
        f"no trajectory '{name}' (looked in {SAVES}). List saves: make rl2-list-demos")


def _dts(t: np.ndarray, n: int) -> np.ndarray:
    """Per-step sleep from recorded timestamps (clamped); fall back to 50 Hz."""
    if t is not None and len(t) == n and n > 1:
        d = np.diff(t)
        d = np.clip(d, 0.0, 0.1)
        return np.append(d, d[-1] if len(d) else 0.02)
    return np.full(n, 1.0 / 50.0)


def replay(name: str, wait_race: bool = True):
    path = _resolve(name)
    d = np.load(path)
    act = d["act"].astype(np.float32)
    t = d["t"] if "t" in d else None
    rec_gate = d["gate_index"] if "gate_index" in d else None
    n = len(act)
    dts = _dts(t, n)
    dur = float(np.sum(dts))
    print(f"[rl2-run] loaded {path}: {n} steps (~{dur:.1f}s recorded)", flush=True)

    sim = SimInterface()
    sim.data["_quiet_vision"] = True
    print("[rl2-run] connected. Start/Restart the race in the sim...", flush=True)
    try:
        # Arm, then wait for the REAL GO -- the GPPilot WAIT_FOR_START gate: a
        # fresh countdown that has ELAPSED with the physics clock live. Waiting on
        # `race_started` alone starts during the countdown and tips the drone over.
        last_arm = 0.0
        if wait_race:
            print("[rl2-run] armed -- WAIT_FOR_START (start/restart the race)...",
                  flush=True)
            go = RaceGo(debug=True)
            while not go(sim.data):
                now = time.time()
                if now - last_arm > 1.0:
                    sim.arm()
                    last_arm = now
                time.sleep(1 / 90.0)
        sim.arm()
        print("[rl2-run] Countdown complete! Replaying recorded actions...", flush=True)

        t0 = time.time()
        for i in range(n):
            if not sim.data.get("armed", False):
                sim.arm()
            roll_deg, pitch_deg, yaw_deg, thrust = controller.action_to_attitude(act[i])
            sim.send_attitude_quat_deg(roll_deg, pitch_deg, yaw_deg, thrust)
            if i % 25 == 0:
                live_gate = sim.data.get("active_gate_index")
                rg = int(rec_gate[i]) if rec_gate is not None else -1
                print(f"  step {i:4d}/{n}  t={time.time()-t0:5.1f}s  "
                      f"recorded_gate={rg}  live_gate={live_gate}  "
                      f"cmd=(r{roll_deg:+.1f} p{pitch_deg:+.1f} y{yaw_deg:+.1f} thr{thrust:.2f})",
                      flush=True)
            time.sleep(float(dts[i]))
        live_gate = sim.data.get("active_gate_index")
        print(f"[rl2-run] replay done. final live gate index = {live_gate}", flush=True)
    except KeyboardInterrupt:
        print("\n[rl2-run] stopped.", flush=True)
    finally:
        sim.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("name", help="trajectory save name (from rl/data/vq2/saves/)")
    ap.add_argument("--no-wait", action="store_true",
                    help="don't wait for race_started; replay immediately")
    args = ap.parse_args()
    replay(args.name, wait_race=not args.no_wait)
