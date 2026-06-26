"""Controller built from MEASURED sim dynamics (see rl.dynamics_id).

Measured facts:
  * Attitude is a pure rate integrator with NO auto-leveling -> must actively
    regulate roll/pitch/yaw ANGLES via rate commands.
  * Hover thrust ~0.27 (very powerful: thrust-accel ~36 m/s^2). [HOVER_T]
  * Yaw-rate command sign is INVERTED (positive cmd -> yaw decreases).

Modes:
  hover  - level attitude + hold spawn altitude (validates the basics).
  course - follow Catmull-Rom raceline through gate_map.json (speed profile + arc projection).

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

from rl import course_path, raceline, spec
from rl.sim_interface import GATE_MAP_PATH, SimInterface
from simulator.transforms import quat_to_yaw

HOVER_T = 0.27
KP_Z, KD_Z = 0.025, 0.030  # thrust is sensitive (accel ~36/unit)
K_ATT = 0.6  # attitude-angle P -> rate command
K_YAW = 0.6  # was 0.4 — rotate onto next gate's heading quicker
# Measured command-sign conventions (pitch normal; roll + yaw inverted).
SIGN_ROLL = -1.0
SIGN_PITCH = +1.0
SIGN_YAW = -1.0
RATE_CLIP = 0.30
YAW_CLIP = 0.8  # was 0.5 — allow faster yaw to cut turn time between gates
LOOK_AHEAD = 4.0
K_CT = 25.4  # cross-track velocity gain onto spline
HZ = 150.0


def rpy(q):
    w, x, y, z = q
    roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = math.asin(max(-1, min(1, 2 * (w * y - z * x))))
    yaw = quat_to_yaw(*q)
    return roll, pitch, yaw


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["hover", "course"], default="course")
    ap.add_argument("--seconds", type=float, default=95.0)
    ap.add_argument("--speed", type=float, default=4.0)  # was 2.8
    ap.add_argument("--lean", type=float, default=0.18, help="max forward lean (rad)")  # was 0.12
    ap.add_argument(
        "--flipz", action="store_true", help="negate gate-map z (climb course)"
    )
    ap.add_argument(
        "--zoff",
        type=float,
        default=-1.0,
        help="altitude offset vs gate center (negative = fly higher, NED)",
    )
    ap.add_argument("--klat", type=float, default=0.11, help="cross-track roll gain")  # was 0.04
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

    gate_map = json.load(open(GATE_MAP_PATH))["gates"]
    n = len(gate_map)
    sim = SimInterface()
    if not sim.wait_for_telemetry():
        print("[f2] no telemetry", flush=True)
        os._exit(1)
    if args.reset:
        sim.reset_sim()
        time.sleep(3)
    # Sync launch to the countdown: wait for the user to hit ENTER at "go".
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

    pts, cum, vprof, total_arc = None, None, None, 0.0
    if args.mode == "course":
        start_pos = np.asarray(s0.pos_ned, float)
        pts, cum = course_path.build_path(start_pos)
        vprof = course_path.build_profile(pts, cum, v_max=args.speed)
        total_arc = cum[-1]
        print(f"[f2] raceline: {len(pts)} pts arc={total_arc:.0f}m", flush=True)

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
        else:
            active = int(sim.data.get("active_gate_index", 0) or 0)
            if active != last_active:
                print(
                    f"[f2] [{time.time() - t0:4.1f}s] ACTIVE GATE -> {active}",
                    flush=True,
                )
                last_active = active
            s = raceline.project_arc(pts, cum, p)
            if active >= n or s >= total_arc - 1.0:
                reason = "COURSE COMPLETE"
                break
            v_set = raceline.speed_at(cum, vprof, s)
            nearest = raceline.point_at_arc(pts, cum, s)
            ahead = raceline.point_at_arc(pts, cum, s + LOOK_AHEAD)
            delta = ahead - nearest
            horizontal = delta[:2]
            hlen = float(np.linalg.norm(horizontal))
            tangent = (
                horizontal / hlen
                if hlen > 1e-6
                else np.array([math.cos(yaw), math.sin(yaw)])
            )
            ct_err = nearest[:2] - p[:2]
            vel_des = v_set * tangent + K_CT * ct_err
            v_horiz = float(np.linalg.norm(vel_des))
            if v_horiz > args.speed:
                vel_des *= args.speed / v_horiz
            bearing = math.atan2(vel_des[1], vel_des[0])
            yaw_err = wrap(bearing - yaw)
            speed = float(np.linalg.norm(v[:2]))
            v_des = float(np.linalg.norm(vel_des))
            align = max(0.0, 1.0 - abs(yaw_err) / 0.5)
            e_cross = ct_err[0] * math.sin(yaw) - ct_err[1] * math.cos(yaw)
            lat_align = max(0.3, 1.0 - abs(e_cross) / 2.5)
            v_cmd = v_des * align * lat_align
            lean = float(np.clip(0.08 * (v_cmd - speed), -0.07, args.lean))
            tgt_pitch = -lean
            tgt_roll = float(np.clip(args.klat * e_cross, -0.16, 0.16))
            tgt_z = (-nearest[2] if args.flipz else nearest[2]) + args.zoff

        # Attitude-angle P -> rate commands, with measured sign conventions.
        roll_cmd = float(
            np.clip(SIGN_ROLL * K_ATT * (tgt_roll - roll), -RATE_CLIP, RATE_CLIP)
        )
        pitch_cmd = float(
            np.clip(SIGN_PITCH * K_ATT * (tgt_pitch - pitch), -RATE_CLIP, RATE_CLIP)
        )
        yaw_cmd = float(np.clip(SIGN_YAW * K_YAW * yaw_err, -YAW_CLIP, YAW_CLIP))
        thrust = float(np.clip(HOVER_T + KP_Z * (z - tgt_z) + KD_Z * vz, 0.18, 0.5))
        sim.send_attitude_rates(roll_cmd, pitch_cmd, yaw_cmd, thrust)

        # Safety.
        gb_z = (spec.quat_to_R(snap.quat).T @ np.array([0.0, 0, 1.0]))[2]
        if gb_z < 0.0:
            reason = "ABORT flipped"
            break
        if z < hold_z - 30 or z > hold_z + 30:
            reason = "ABORT altitude"
            break

        now = time.time() - t0
        if now - last_log >= 0.5:
            extra = ""
            if args.mode == "course" and pts is not None:
                s_log = raceline.project_arc(pts, cum, p)
                extra = f" s={s_log:5.1f}/{total_arc:.0f}"
            print(
                f"[f2] [{now:4.1f}s] rpy=({math.degrees(roll):+4.0f},"
                f"{math.degrees(pitch):+4.0f},{math.degrees(yaw):+4.0f}) "
                f"z={z:+5.1f} v=({v[0]:+4.1f},{v[1]:+4.1f},{v[2]:+4.1f}) thr={thrust:.2f}{extra}",
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
