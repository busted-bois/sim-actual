"""Controller built from MEASURED sim dynamics (see rl.dynamics_id).

Measured facts:
  * Attitude is a pure rate integrator with NO auto-leveling -> must actively
    regulate roll/pitch/yaw ANGLES via rate commands.
  * Hover thrust ~0.27 (very powerful: thrust-accel ~36 m/s^2). [HOVER_T]
  * Yaw-rate command sign is INVERTED (positive cmd -> yaw decreases).

Modes:
  hover  - level attitude + hold spawn altitude (validates the basics).
  course - fly the gate map, gates advanced by sim active_gate_index.

    uv run -m rl.fly2 --mode hover  --seconds 6
    uv run -m rl.fly2 --mode course --seconds 60 --speed 3
"""

import argparse
import json
import math
import os
import sys
import time

import numpy as np

from rl import spec
from rl.fly2_course import (
    HOVER_T,
    Fly2Config,
    K_ATT,
    K_YAW,
    KD_Z,
    KP_Z,
    RATE_CLIP,
    SIGN_PITCH,
    SIGN_ROLL,
    SIGN_YAW,
    YAW_CLIP,
    compute_course_rates,
    rpy,
    wrap,
)
from rl.sim_interface import GATE_MAP_PATH, SimInterface

HZ = 150.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["hover", "course"], default="course")
    ap.add_argument("--seconds", type=float, default=95.0)
    ap.add_argument("--speed", type=float, default=2.8)
    ap.add_argument("--lean", type=float, default=0.12, help="max forward lean (rad)")
    ap.add_argument(
        "--flipz", action="store_true", help="negate gate-map z (climb course)"
    )
    ap.add_argument(
        "--zoff",
        type=float,
        default=-1.0,
        help="altitude offset vs gate center (negative = fly higher, NED)",
    )
    ap.add_argument("--klat", type=float, default=0.04, help="cross-track roll gain")
    ap.add_argument(
        "--no-wait",
        dest="wait",
        action="store_false",
        help="launch immediately instead of waiting for ENTER",
    )
    ap.add_argument(
        "--reset",
        action="store_true",
        help="send a sim reset before launching (else rely on race restart)",
    )
    args = ap.parse_args()

    cfg = Fly2Config(
        speed=args.speed,
        lean=args.lean,
        klat=args.klat,
        zoff=args.zoff,
        flipz=args.flipz,
    )
    gate_map = json.load(open(GATE_MAP_PATH))["gates"]
    n = len(gate_map)
    sim = SimInterface()
    if not sim.wait_for_telemetry():
        print("[f2] no telemetry", flush=True)
        os._exit(1)
    if args.reset:
        sim.reset_sim()
        time.sleep(3)
    if args.wait and args.mode == "course":
        try:
            input("[f2] READY -- press ENTER the moment the countdown hits 0...")
        except EOFError:
            pass
    sim.arm()
    time.sleep(0.2)
    s0 = sim.snapshot()
    hold_z = s0.pos_ned[2]
    hold_yaw = rpy(s0.quat)[2]
    print(f"[f2] mode={args.mode} hold_z={hold_z:.1f} hover_t={HOVER_T}", flush=True)

    t0 = time.time()
    last_log = 0.0
    last_active = -1
    reason = "timeout"
    while time.time() - t0 < args.seconds:
        snap = sim.snapshot()
        if not snap.has_pose():
            time.sleep(1 / HZ)
            continue
        p = np.asarray(snap.pos_ned, float)
        v = np.asarray(snap.vel_ned, float)
        roll, pitch, yaw = rpy(snap.quat)
        z, vz = p[2], v[2]

        if args.mode == "hover":
            tgt_pitch, tgt_roll, yaw_err, tgt_z = 0.0, 0.0, wrap(hold_yaw - yaw), hold_z
            roll_cmd = float(
                np.clip(SIGN_ROLL * K_ATT * (tgt_roll - roll), -RATE_CLIP, RATE_CLIP)
            )
            pitch_cmd = float(
                np.clip(SIGN_PITCH * K_ATT * (tgt_pitch - pitch), -RATE_CLIP, RATE_CLIP)
            )
            yaw_cmd = float(np.clip(SIGN_YAW * K_YAW * yaw_err, -YAW_CLIP, YAW_CLIP))
            thrust = float(np.clip(HOVER_T + KP_Z * (z - tgt_z) + KD_Z * vz, 0.18, 0.5))
        else:
            active = int(sim.data.get("active_gate_index", 0) or 0)
            if active != last_active:
                print(
                    f"[f2] [{time.time() - t0:4.1f}s] ACTIVE GATE -> {active}",
                    flush=True,
                )
                last_active = active
            if active >= n:
                reason = "COURSE COMPLETE"
                break
            roll_cmd, pitch_cmd, yaw_cmd, thrust = compute_course_rates(
                p, v, snap.quat, active, gate_map, hold_z, cfg
            )

        sim.send_attitude_rates(roll_cmd, pitch_cmd, yaw_cmd, thrust)

        gb_z = (spec.quat_to_R(snap.quat).T @ np.array([0.0, 0, 1.0]))[2]
        if gb_z < 0.0:
            reason = "ABORT flipped"
            break
        if z < hold_z - 30 or z > hold_z + 30:
            reason = "ABORT altitude"
            break

        now = time.time() - t0
        if now - last_log >= 0.5:
            print(
                f"[f2] [{now:4.1f}s] rpy=({math.degrees(roll):+4.0f},"
                f"{math.degrees(pitch):+4.0f},{math.degrees(yaw):+4.0f}) "
                f"z={z:+5.1f} v=({v[0]:+4.1f},{v[1]:+4.1f},{v[2]:+4.1f}) thr={thrust:.2f}",
                flush=True,
            )
            last_log = now
        time.sleep(1 / HZ)

    sim.send_attitude_rates(0, 0, 0, HOVER_T)
    print(
        f"[f2] === DONE {reason} final={np.round(sim.snapshot().pos_ned, 1)} "
        f"active={sim.data.get('active_gate_index')} ===",
        flush=True,
    )
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
