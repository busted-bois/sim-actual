"""Thrust-response characterization of the real sim (translational env calib).

The rotation-calibrated policy no longer tumbles, but it flies ~37 m/s and
overshoots gates: the env's TRANSLATIONAL model (thrust->accel) is still an
assumption. This sweeps thrust at zero rates from a reset (level) hover and logs
vertical velocity, to measure hover thrust and thrust->accel. Vertical accel
(NED, +down) = gravity - thrust_accel * thrust, so fitting accel vs thrust gives
thrust_accel and the hover point. Drag is then tuned so the env's terminal speed
matches the ~37 m/s seen on deploy.

    uv run -m rl.thrust_id
"""

import os
import sys
import time

import numpy as np

from rl.sim_interface import SimInterface

HZ = 100.0


def measure(sim, thrust, secs=1.5):
    sim.reset_sim()
    time.sleep(3)
    sim.arm()
    time.sleep(0.3)
    t0 = time.time()
    nxt = 0.0
    ts, vzs = [], []
    while time.time() - t0 < secs:
        sim.send_attitude_rates(0, 0, 0, thrust)
        snap = sim.snapshot()
        vz = snap.vel_ned[2] if snap.vel_ned else 0.0
        z = snap.pos_ned[2] if snap.pos_ned else 0.0
        el = time.time() - t0
        ts.append(el)
        vzs.append(vz)
        if el >= nxt:
            print(f"[thr] cmd={thrust:.2f} t={el:.2f} vz={vz:+.2f} z={z:+.1f}", flush=True)
            nxt += 0.3
        time.sleep(1 / HZ)
    accel = float(np.polyfit(ts, vzs, 1)[0]) if len(ts) > 10 else 0.0
    print(
        f"[thr] >>> cmd={thrust:.2f} -> vert accel {accel:+.2f} m/s^2 "
        f"(NED +down; <0 = climbs, ~0 = hover)",
        flush=True,
    )
    return accel


def main():
    sim = SimInterface()
    if not sim.wait_for_telemetry():
        print("[thr] no telemetry — is the simulator running?", flush=True)
        os._exit(1)
    levels = (0.20, 0.27, 0.35, 0.45)
    accels = [measure(sim, t) for t in levels]
    # accel = g - thrust_accel*thrust  =>  slope of accel-vs-thrust = -thrust_accel
    slope = float(np.polyfit(levels, accels, 1)[0])
    thrust_accel = -slope
    hover = float(np.interp(0.0, accels[::-1], list(levels)[::-1])) if accels else 0.0
    print(
        f"[thr] === thrust_accel ~ {thrust_accel:.1f} m/s^2  hover ~ {hover:.3f} "
        f"(env assumes accel~36, hover~0.27) ===",
        flush=True,
    )
    sim.send_attitude_rates(0, 0, 0, 0.27)
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
