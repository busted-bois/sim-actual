"""Rate-response characterization of the real sim (for RL env calibration).

The RL policy tumbles on deploy because the sim's rotational response is far
hotter than the training model: the policy commands up to 4 rad/s and the drone
hits ~13 rad/s, while fly2 flies stably clipping rate commands to 0.3 rad/s.

This probe applies CONSTANT rate-command steps of increasing magnitude on one
axis at a time and logs the ACTUAL body rate (and angle) the sim reaches, to
measure the command -> body-rate mapping (gain, linearity, sign). With that, the
env (rl.env) rotational model + MAX_*_RATE can be rebuilt to match before
retraining. Raw commands (no SIGN_* applied) so this reads the plant directly.

    uv run -m rl.rate_id
"""

import math
import os
import sys
import time

import numpy as np

from rl.sim_interface import SimInterface
from simulator.transforms import quat_to_yaw

HZ = 100.0
AXES = {0: "roll", 1: "pitch", 2: "yaw"}


def angle(snap, axis):
    w, x, y, z = snap.quat
    if axis == 0:
        return math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    if axis == 1:
        return math.asin(max(-1, min(1, 2 * (w * y - z * x))))
    return quat_to_yaw(*snap.quat)


def measure(sim, axis, mag, secs=1.2):
    cmd = [0.0, 0.0, 0.0]
    cmd[axis] = mag
    sim.reset_sim()
    time.sleep(3)
    sim.arm()
    time.sleep(0.3)
    t0 = time.time()
    nxt = 0.0
    ws = []
    while time.time() - t0 < secs:
        sim.send_attitude_rates(cmd[0], cmd[1], cmd[2], 0.27)
        snap = sim.snapshot()
        w = snap.ang_vel if snap.ang_vel else [0.0, 0.0, 0.0]
        ws.append(w[axis])
        el = time.time() - t0
        if el >= nxt:
            print(
                f"[rate] {AXES[axis]} cmd={mag:+.2f} t={el:.2f} "
                f"w={np.round(w, 2)} ang={math.degrees(angle(snap, axis)):+6.1f}d",
                flush=True,
            )
            nxt += 0.2
        time.sleep(1 / HZ)
    ss = np.array(ws[len(ws) // 2 :])  # steady-state ~ second half
    mean = float(ss.mean()) if len(ss) else 0.0
    gain = mean / mag if mag else 0.0
    print(
        f"[rate] >>> {AXES[axis]} cmd={mag:+.2f} -> actual w~{mean:+.2f} rad/s "
        f"(gain {gain:+.2f}, sign {'+' if mean * mag > 0 else '-'})",
        flush=True,
    )


def main():
    sim = SimInterface()
    if not sim.wait_for_telemetry():
        print("[rate] no telemetry — is the simulator running?", flush=True)
        os._exit(1)
    for axis in (1, 0, 2):  # pitch, roll, yaw
        for mag in (0.3, 1.0, 2.0):
            print(f"[rate] === {AXES[axis]} step cmd={mag} ===", flush=True)
            measure(sim, axis, mag)
    sim.send_attitude_rates(0, 0, 0, 0.27)
    print("[rate] done", flush=True)
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
