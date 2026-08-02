"""Controller built from MEASURED sim dynamics (see rl.dynamics_id).

Measured facts:
  * Attitude is a pure rate integrator with NO auto-leveling -> must actively
    regulate roll/pitch/yaw ANGLES via rate commands.
  * Hover thrust ~0.27 (very powerful: thrust-accel ~36 m/s^2). [HOVER_T]
  * Yaw-rate command sign is INVERTED (positive cmd -> yaw decreases).

Modes:
  hover  - level attitude + hold spawn altitude (validates the basics).
  course - fly the gate map, gates advanced by sim active_gate_index.

    uv run -m rl.experts.fly2 --mode hover  --seconds 6
    uv run -m rl.experts.fly2 --mode course --seconds 60 --speed 3
"""

import argparse
import csv
import json
import math
import os
import sys
import time

import numpy as np

from rl.core import spec
from rl.experts.fly2_course import (
    EST_SIGNS,
    HOVER_T,
    Fly2Config,
    compute_course_rates,
    rates_from_attitude_targets,
    rpy,
    wrap,
)
from rl.environment.sim_interface import GATE_MAP_PATH, SimInterface
from simulator import display
from simulator.vision_nav import VisionGuidance, VisualServo

HZ = 90.0  # spec VADR-TS-003 4.4: command rate must stay < 100 Hz


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["hover", "course", "vision"], default="course")
    ap.add_argument("--seconds", type=float, default=95.0)
    ap.add_argument(
        "--gates", type=int, default=6, help="vision mode: stop after N gates passed"
    )
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
    ap.add_argument(
        "--est",
        action="store_true",
        help="fly on the IMU+vision state estimator even if odometry is present "
        "(VQ2 dress rehearsal; odometry then only feeds the shadow CSV)",
    )
    args = ap.parse_args()

    cfg = Fly2Config(
        speed=args.speed,
        lean=args.lean,
        klat=args.klat,
        zoff=args.zoff,
        flipz=args.flipz,
    )
    # Vision mode flies purely from detected gates -- no hardcoded map.
    gate_map = []
    n = 0
    if args.mode == "course":
        gate_map = json.load(open(GATE_MAP_PATH))["gates"]
        n = len(gate_map)
    guide = None  # constructed after est_mode is known (speed differs)
    sim = SimInterface(use_estimator=args.est)
    if not sim.wait_for_telemetry():
        print("[f2] no telemetry", flush=True)
        os._exit(1)
    if args.reset:
        sim.reset_sim()
        time.sleep(3)
        # The teleport invalidates the gyro-integrated attitude: re-init.
        sim.estimator.reset()
        time.sleep(1.5)
    # Sync launch to the countdown: wait for the user to hit ENTER at "go".
    if args.wait and args.mode in ("course", "vision"):
        try:
            input("[f2] READY -- press ENTER the moment the countdown hits 0...")
        except EOFError:
            pass
    sim.arm()
    time.sleep(0.2)
    s0 = sim.snapshot()
    hold_z = s0.pos_ned[2]
    hold_yaw = rpy(s0.quat)[2]
    # Estimator-driven flight needs the gyro-frame command signs.
    est_mode = args.est or sim.data.get("odometry") is None
    att_signs = EST_SIGNS if est_mode else None
    if args.mode == "vision":
        # Estimator regime: body-frame visual servoing -- world-position
        # drift cannot create phantom targets (measured failure of the
        # map-based guidance without odometry). Training/odometry keeps the
        # original world-map guidance.
        guide = VisualServo() if est_mode else VisionGuidance()
    print(
        f"[f2] mode={args.mode} hold_z={hold_z:.1f} hover_t={HOVER_T}"
        f" est_mode={est_mode}",
        flush=True,
    )

    # Per-run nav telemetry: guidance phase + axis-frame errors at 10 Hz.
    nav_log, nav_wr, last_nav_t, min_gate_d = None, None, 0.0, None
    if args.mode == "vision":
        os.makedirs(os.path.join("rl", "data"), exist_ok=True)
        nav_path = os.path.join(
            "rl", "data", time.strftime("nav_log_%Y%m%d_%H%M%S.csv")
        )
        nav_log = open(nav_path, "w", newline="")
        nav_wr = csv.writer(nav_log)
        nav_wr.writerow(
            "t phase s lat vert dist pn pe pd vn ve vd status n_passed".split()
        )
        print(f"[f2] nav log -> {nav_path}", flush=True)

    display.start()  # live vision window (what the drone's camera sees)

    t0 = time.time()
    last_log = 0.0
    last_active = -1
    last_shown_tag = None
    last_pose_fid = None
    last_col = sim.data.get("last_collision")  # ignore stale pre-run hits
    last_col_t = -1e9
    scan_since = None
    reason = "timeout"
    while time.time() - t0 < args.seconds:
        # Pump the vision window whenever a new (detected) frame is ready.
        img, tag = display.pick(sim.data)
        if tag is not None and tag != last_shown_tag:
            last_shown_tag = tag
            display.tick(img, time.time() - t0)

        snap = sim.snapshot()
        if not snap.has_pose():
            time.sleep(1 / HZ)
            continue
        p = np.asarray(snap.pos_ned, float)
        v = np.asarray(snap.vel_ned, float)
        roll, pitch, yaw = rpy(snap.quat)
        z, vz = p[2], v[2]

        vstatus = ""
        if args.mode == "hover":
            roll_cmd, pitch_cmd, yaw_cmd, thrust = rates_from_attitude_targets(
                roll,
                pitch,
                z,
                vz,
                0.0,
                0.0,
                wrap(hold_yaw - yaw),
                hold_z,
                signs=att_signs,
            )
        elif args.mode == "vision" and time.time() - t0 < 1.2:
            # TAKEOFF: level out and climb clear of the spawn ramp.
            roll_cmd, pitch_cmd, yaw_cmd, thrust = rates_from_attitude_targets(
                roll, pitch, z, vz, 0.0, 0.0, 0.0, hold_z - 2.0, signs=att_signs
            )
            vstatus = "TAKEOFF"
        elif args.mode == "vision":
            pose_data = sim.data.get("pose")
            fresh = pose_data is not None and pose_data["frame_id"] != last_pose_fid
            if fresh:
                last_pose_fid = pose_data["frame_id"]
            gates = pose_data["gates"] if fresh else []
            cmd = guide.update(gates, p, v, snap.quat, yaw, time.time())
            if est_mode and fresh:
                R_wb = spec.quat_to_R(snap.quat)
                for map_p, gate_body in guide.last_matches:
                    sim.estimator.update_landmark(map_p - R_wb @ gate_body)
            vstatus = cmd.status
            if nav_wr is not None and time.time() - last_nav_t >= 0.1:
                last_nav_t = time.time()
                dbg = getattr(guide, "debug", None) or {}
                d = dbg.get("dist")
                if d is not None:
                    min_gate_d = d if min_gate_d is None else min(min_gate_d, d)
                nav_wr.writerow(
                    [
                        round(time.time() - t0, 2),
                        dbg.get("phase", ""),
                        *(
                            None if x is None else round(float(x), 2)
                            for x in (dbg.get("s"), dbg.get("lat"), dbg.get("vert"), d)
                        ),
                        *(round(float(x), 2) for x in (*p, *v)),
                        cmd.status,
                        guide.n_passed,
                    ]
                )
            col = sim.data.get("last_collision")
            if col is not None and col != last_col:
                last_col = col
                if time.time() - last_col_t > 1.0:
                    last_col_t = time.time()
                    if est_mode:
                        sim.estimator.notify_collision()
                    print(f"[f2] [{time.time() - t0:4.1f}s] COLLISION", flush=True)
            if cmd.status.startswith("SCAN"):
                if scan_since is None:
                    scan_since = time.time()
            else:
                scan_since = None
            if guide.n_passed >= args.gates:
                reason = "COURSE COMPLETE (vision)"
                break
            roll_cmd, pitch_cmd, yaw_cmd, thrust = rates_from_attitude_targets(
                roll,
                pitch,
                z,
                vz,
                cmd.tgt_roll,
                cmd.tgt_pitch,
                cmd.yaw_err,
                cmd.tgt_z,
                signs=att_signs,
            )
            # Vision mode: bound vertical authority hard.
            z_err = float(np.clip(z - cmd.tgt_z, -3.0, 3.0))
            vz_c = float(np.clip(vz, -4.0, 4.0))
            thrust = float(
                np.clip(HOVER_T + 0.025 * z_err + 0.030 * vz_c, 0.20, 0.36)
            )
            if scan_since is not None and time.time() - scan_since > 1.5:
                thrust = HOVER_T - 0.012
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
        z_abort = 60 if est_mode else 30
        if z < hold_z - z_abort or z > hold_z + z_abort:
            reason = "ABORT altitude"
            break

        now = time.time() - t0
        if now - last_log >= 0.5:
            print(
                f"[f2] [{now:4.1f}s] rpy=({math.degrees(roll):+4.0f},"
                f"{math.degrees(pitch):+4.0f},{math.degrees(yaw):+4.0f}) "
                f"z={z:+5.1f} v=({v[0]:+4.1f},{v[1]:+4.1f},{v[2]:+4.1f}) thr={thrust:.2f}"
                f"{('  ' + vstatus) if vstatus else ''}",
                flush=True,
            )
            last_log = now
        time.sleep(1 / HZ)

    sim.send_attitude_rates(0, 0, 0, HOVER_T)
    display.close()
    if nav_log is not None:
        nav_log.close()
        print(
            f"[f2] nav summary: passed={guide.n_passed if guide else 0} "
            f"closest_gate={min_gate_d if min_gate_d is not None else 'n/a'}",
            flush=True,
        )
    print(
        f"[f2] === DONE {reason} final={np.round(sim.snapshot().pos_ned, 1)} "
        f"active={sim.data.get('active_gate_index')} ===",
        flush=True,
    )
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
