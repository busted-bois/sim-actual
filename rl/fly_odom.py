"""Fly the real course: odometry + gate map, pilot.py-style control.

The real sim has an inner attitude loop — a constant pitch-RATE setpoint
yields a bounded forward lean (the existing HSV pilot relies on this). So we
do NOT use the pure-integrator geometric controller here. Instead we reuse the
proven control structure from simulator/pilot.py:

  * altitude PID on thrust (targets the active gate's altitude),
  * yaw-rate toward the gate bearing,
  * gentle forward lean (pitch-rate) scaled by heading alignment,

but driven by ground-truth odometry + the captured gate map, with gate
progression from the sim's active_gate_index.

    uv run -m rl.fly_odom --seconds 15            # supervised test
    uv run -m rl.fly_odom --seconds 120 --cruise -0.22
"""

import argparse
import json
import math
import os
import sys
import time

import numpy as np

from rl import spec
from rl.sim_interface import GATE_MAP_PATH, SimInterface
from simulator.transforms import quat_to_yaw

# Proven gains from simulator/pilot.py.
ALT_TRIM = 0.55
KP_Z, KI_Z, KD_Z = 0.15, 0.01, 0.20
KP_YAW = 1.0
MAX_YAW = 1.5
DT = 1 / 200.0


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=15.0)
    ap.add_argument("--speed", type=float, default=4.0, help="target forward speed m/s")
    ap.add_argument(
        "--align", type=float, default=0.35, help="yaw err (rad) to allow full lean"
    )
    args = ap.parse_args()

    gate_map = json.load(open(GATE_MAP_PATH))["gates"]
    n = len(gate_map)
    z_abort = max(g["pos"][2] for g in gate_map) + 8.0

    sim = SimInterface()
    if not sim.wait_for_telemetry():
        print("[fly] no telemetry", flush=True)
        os._exit(1)
    sim.arm()
    print(
        f"[fly] armed; {n} gates, z_abort={z_abort:.1f}, target_speed={args.speed}",
        flush=True,
    )
    # Speed regulation params — keep speed bounded to avoid the fly-away quirk.
    KP_V = 0.06  # pitch-rate per m/s speed error
    MAX_LEAN = 0.16  # max forward lean pitch-rate
    MAX_BRAKE = 0.25  # max backward (braking) pitch-rate
    HARD_SPEED = 7.0  # above this, force hard brake regardless

    z_int = 0.0
    last_active = -1
    last_log = 0.0
    reason = "timeout"
    t0 = time.time()
    while time.time() - t0 < args.seconds:
        snap = sim.snapshot()
        if not snap.has_pose():
            time.sleep(DT)
            continue
        p = np.asarray(snap.pos_ned, float)
        v = np.asarray(snap.vel_ned, float)
        q = np.asarray(snap.quat, float)
        z, vz = p[2], v[2]
        yaw = quat_to_yaw(*q)

        active = int(sim.data.get("active_gate_index", 0) or 0)
        if active != last_active:
            print(
                f"[fly] [{time.time() - t0:5.1f}s] ACTIVE GATE -> {active}", flush=True
            )
            last_active = active
        if active >= n:
            reason = "COURSE COMPLETE"
            break
        target = np.asarray(gate_map[active]["pos"], float)

        dx, dy, dz = target - p
        dist = math.hypot(dx, dy)
        bearing = math.atan2(dy, dx)
        yaw_err = wrap(bearing - yaw)
        yaw_rate = float(np.clip(KP_YAW * yaw_err, -MAX_YAW, MAX_YAW))

        # Speed-regulated forward lean. Target speed shrinks when off-heading or
        # near the gate, so we pass through cleanly instead of overshooting.
        speed = float(np.linalg.norm(v))
        align = max(0.0, 1.0 - abs(yaw_err) / args.align)
        v_des = args.speed * align * min(1.0, 0.4 + dist / 12.0)
        # pitch-rate: lean forward to reach v_des, brake (positive) if too fast.
        pitch = float(np.clip(-KP_V * (v_des - speed), -MAX_LEAN, MAX_BRAKE))
        if speed > HARD_SPEED:
            pitch = MAX_BRAKE  # safety: hard brake to prevent fly-away

        # Altitude PID on thrust toward the gate's altitude.
        err = z - target[2]
        z_int = float(np.clip(z_int + err * DT, -0.5, 0.5))
        thrust = float(
            np.clip(ALT_TRIM + KP_Z * err + KI_Z * z_int + KD_Z * vz, 0.0, 1.0)
        )

        sim.send_attitude_rates(0.0, pitch, yaw_rate, thrust)

        gb_z = (spec.quat_to_R(q).T @ np.array([0.0, 0, 1.0]))[2]
        if gb_z < -0.2:
            reason = "ABORT flipped"
            break
        if z > z_abort:
            reason = "ABORT below-course"
            break

        now = time.time() - t0
        if now - last_log >= 1.0:
            print(
                f"[fly] [{now:5.1f}s] g{active} pos=({p[0]:6.1f},{p[1]:5.1f},{z:6.1f}) "
                f"dist={dist:5.1f} yaw_err={math.degrees(yaw_err):+5.0f}d "
                f"thr={thrust:.2f} spd={np.linalg.norm(v):4.1f}",
                flush=True,
            )
            last_log = now
        time.sleep(DT)

    sim.send_attitude_rates(0, 0, 0, ALT_TRIM)
    snap = sim.snapshot()
    print(
        f"[fly] === DONE: {reason} | final pos={np.round(snap.pos_ned, 1)} "
        f"active_gate={sim.data.get('active_gate_index')} ===",
        flush=True,
    )
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
