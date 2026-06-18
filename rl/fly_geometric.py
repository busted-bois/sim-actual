"""Fly the real course with the geometric controller + odometry + gate map.

Round-1 deploy: the sim gives absolute odometry and the gate map, and reports
active_gate_index (ground-truth current target). We feed odometry pose to the
geometric controller, target the active gate, and send attitude-rate+thrust.
No vision, no learned policy — fastest reliable path.

Safety: aborts on flip / ground / out-of-bounds, and after --seconds.

    uv run -m rl.fly_geometric --seconds 15 --speed 3      # short supervised test
    uv run -m rl.fly_geometric --seconds 90 --speed 6      # full course attempt
"""

import argparse
import json
import os
import sys
import time

import numpy as np

from rl import spec
from rl.control import geometric_action
from rl.sim_interface import GATE_MAP_PATH, SimInterface


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=15.0)
    ap.add_argument("--speed", type=float, default=3.0, help="max approach speed m/s")
    ap.add_argument("--kp_att", type=float, default=6.0)
    args = ap.parse_args()

    gate_map = json.load(open(GATE_MAP_PATH))["gates"]
    n_gates = len(gate_map)
    sim = SimInterface()
    if not sim.wait_for_telemetry():
        print("no telemetry", flush=True)
        os._exit(1)

    sim.arm()
    print(
        f"armed; flying with geometric controller, {n_gates} gates, "
        f"max {args.seconds:.0f}s, speed={args.speed}",
        flush=True,
    )

    t0 = time.time()
    last_log = 0.0
    last_active = -1
    reason = "timeout"
    hz = 200.0
    while time.time() - t0 < args.seconds:
        snap = sim.snapshot()
        if not snap.has_pose():
            time.sleep(1 / hz)
            continue
        p = np.asarray(snap.pos_ned, float)
        v = np.asarray(snap.vel_ned, float)
        q = np.asarray(snap.quat, float)

        active = int(sim.data.get("active_gate_index", 0) or 0)
        if active != last_active:
            print(f"[{time.time() - t0:5.1f}s] ACTIVE GATE -> {active}", flush=True)
            last_active = active
        if active >= n_gates:
            reason = "COURSE COMPLETE"
            break

        target = np.asarray(gate_map[active]["pos"], float)
        a = geometric_action(p, v, q, target, kp_att=args.kp_att, max_speed=args.speed)
        roll, pitch, yaw, thrust = spec.scale_action(a)
        sim.send_attitude_rates(roll, pitch, yaw, thrust)

        # Safety: flipped (gravity points up in body) or hit ground.
        gb_z = (spec.quat_to_R(q).T @ np.array([0.0, 0, 1.0]))[2]
        if gb_z < -0.2:
            reason = "ABORT flipped"
            break
        if p[2] > 1.0:
            reason = "ABORT ground"
            break

        now = time.time() - t0
        if now - last_log >= 1.0:
            dist = float(np.linalg.norm(target - p))
            spd = float(np.linalg.norm(v))
            print(
                f"[{now:5.1f}s] gate{active} pos=({p[0]:6.1f},{p[1]:5.1f},{p[2]:6.1f}) "
                f"dist={dist:5.1f} spd={spd:4.1f} thr={thrust:.2f}",
                flush=True,
            )
            last_log = now
        time.sleep(1 / hz)

    # Gentle hover command on exit.
    sim.send_attitude_rates(0, 0, 0, spec.HOVER_THRUST)
    snap = sim.snapshot()
    print(
        f"=== DONE: {reason} | final pos={np.round(snap.pos_ned, 1)} "
        f"active_gate={sim.data.get('active_gate_index')} ===",
        flush=True,
    )
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
