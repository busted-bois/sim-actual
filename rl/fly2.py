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
from rl.sim_interface import GATE_MAP_PATH, SimInterface
from simulator import display
from simulator.transforms import quat_to_yaw
from simulator.vision_nav import VisionGuidance, VisualServo

HOVER_T = 0.27
# Thrust is sensitive (accel ~36/unit). KD tuned by camera evidence
# 2026-07-05: at 0.030 the drone crossed gate 1 centred but ballooned
# +1.4 m after it and hit the structure above; at 0.055 it flew low and
# clipped the bottom bar on entry. 0.040 splits the corridor.
KP_Z, KD_Z = 0.025, 0.040
K_ATT = 0.6  # attitude-angle P -> rate command
K_YAW = 0.4
# Measured command-sign conventions vs ODOMETRY attitude (pitch normal;
# roll + yaw inverted). Tuned in Training mode.
SIGN_ROLL = -1.0
SIGN_PITCH = +1.0
SIGN_YAW = -1.0
# Vs the GYRO-integrated estimator attitude the plant is inverted on ALL
# axes (measured live 2026-07-01: cmd +0.2 -> gyro ~-0.48 on each axis).
EST_SIGNS = (-1.0, -1.0, -1.0)
RATE_CLIP = 0.30
YAW_CLIP = 0.5
HZ = 90.0  # spec VADR-TS-003 4.4: command rate must stay < 100 Hz


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
        "(VQ2 dress rehearsal; odometry then only feeds the shadow report)",
    )
    ap.add_argument(
        "--no-record",
        dest="record",
        action="store_false",
        help="disable the replayable flight event log (rl/data/est_log_*.jsonl)",
    )
    args = ap.parse_args()

    # Vision mode flies purely from detected gates -- no hardcoded map.
    gate_map = []
    n = 0
    if args.mode == "course":
        gate_map = json.load(open(GATE_MAP_PATH))["gates"]
        n = len(gate_map)
    guide = None  # constructed after est_mode is known (speed differs)
    # Vision flights always record (a Training-mode flight is a truth-labeled
    # log -- gold for offline replay); other modes record iff estimator-driven.
    if not args.record:
        record = False
    elif args.mode == "vision":
        record = True
    else:
        record = None  # auto: record iff the estimator drives flight
    sim = SimInterface(use_estimator=args.est, record=record)
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
    s_roll, s_pitch, s_yaw = (
        EST_SIGNS if est_mode else (SIGN_ROLL, SIGN_PITCH, SIGN_YAW)
    )
    if args.mode == "vision":
        # Estimator pose is coarser than odometry: approach slower so the
        # aim tolerance at the opening is larger.
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
        # Pump the vision window whenever a new (detected) frame is ready, so it
        # stays responsive without throttling the 150 Hz control loop below.
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
        takeoff_boost = False
        if args.mode == "hover":
            tgt_pitch, tgt_roll, yaw_err, tgt_z = 0.0, 0.0, wrap(hold_yaw - yaw), hold_z
        elif args.mode == "vision" and time.time() - t0 < 2.2:
            # TAKEOFF: the drone spawns nose-down on a ramp and coasts to
            # gate 1 (~5 m) by ~2.5 s NO MATTER WHAT (measured across 10
            # flights 2026-07-05) -- the only free variable is its state on
            # arrival. Counter-hold kills the ramp slide (works: 0.1 m/s at
            # the gate) and a thrust BOOST completes the ~2.5 m climb in
            # time (the plain z-PD only managed 1.3 m -> parked on the
            # gate's bottom panel). Then settle so the crossing carries no
            # vertical momentum (climb-throughs exited the top of the 1.5 m
            # opening and clipped the frame at z~-3.6 every flight).
            tt = time.time() - t0
            tgt_roll, tgt_pitch, yaw_err, tgt_z = 0.0, 0.10, 0.0, hold_z - 2.6
            takeoff_boost = tt < 1.4
            vstatus = "TAKEOFF"
        elif args.mode == "vision":
            # YOLO runs at 3-10 Hz; the 150 Hz loop must only ingest each
            # pose frame ONCE -- re-feeding a stale detection against the
            # moving estimate corrupts both the map and the landmark fixes.
            pose_data = sim.data.get("pose")
            fresh = pose_data is not None and pose_data["frame_id"] != last_pose_fid
            if fresh:
                last_pose_fid = pose_data["frame_id"]
            gates = pose_data["gates"] if fresh else []
            cmd = guide.update(gates, p, v, snap.quat, yaw, time.time())
            # Re-observed confirmed gates are landmarks: p = map - R_wb @ body.
            # Only when the estimator drives the pose -- else the map lives in
            # the odometry frame and would corrupt the boot-anchored filter.
            if est_mode and fresh:
                R_wb = spec.quat_to_R(snap.quat)
                for map_p, gate_body in guide.last_matches:
                    # raw provenance lets replay re-derive p_meas from a
                    # REPLAYED attitude (est_replay --rederive-landmarks).
                    sim.estimator.update_landmark(
                        map_p - R_wb @ gate_body,
                        raw={
                            "map_p": map_p,
                            "gate_body": gate_body,
                            "quat_used": snap.quat,
                        },
                    )
            tgt_roll, tgt_pitch, yaw_err, tgt_z = (
                cmd.tgt_roll,
                cmd.tgt_pitch,
                cmd.yaw_err,
                cmd.tgt_z,
            )
            vstatus = cmd.status
            # A collision kicks the real state; let the estimator re-anchor.
            # Scrapes emit collisions at 60 Hz -- treat them as ONE episode
            # (repeated covariance inflation would blow the filter open).
            col = sim.data.get("last_collision")
            if col is not None and col != last_col:
                last_col = col
                if time.time() - last_col_t > 1.0:
                    last_col_t = time.time()
                    if est_mode:
                        sim.estimator.notify_collision()
                    print(f"[f2] [{time.time() - t0:4.1f}s] COLLISION", flush=True)
            # Sim race status survives VQ2 -- active_gate_index is the
            # AUTHORITATIVE pass count; log it so flight logs grade the
            # servo's own pass detection (guide.n_passed) against truth.
            active = int(sim.data.get("active_gate_index", 0) or 0)
            if active != last_active:
                print(
                    f"[f2] [{time.time() - t0:4.1f}s] SIM GATE -> {active} "
                    f"(vision counted {guide.n_passed})",
                    flush=True,
                )
                last_active = active
            # Track how long we've been scanning with no confirmed gate.
            if cmd.status.startswith("SCAN"):
                if scan_since is None:
                    scan_since = time.time()
            else:
                scan_since = None
            if guide.n_passed >= args.gates:
                reason = "COURSE COMPLETE (vision)"
                break
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
            g = np.asarray(gate_map[active]["pos"], float)
            dx, dy = g[0] - p[0], g[1] - p[1]
            dist = math.hypot(dx, dy)
            bearing = math.atan2(dy, dx)
            yaw_err = wrap(bearing - yaw)
            speed = float(np.linalg.norm(v[:2]))
            align = max(0.0, 1.0 - abs(yaw_err) / 0.4)
            v_des = args.speed * align * min(1.0, 0.3 + dist / 10.0)
            # forward (toward gate) = NEGATIVE pitch (measured); brake = positive.
            lean = float(np.clip(0.05 * (v_des - speed), -0.05, args.lean))
            tgt_pitch = -lean
            # Cross-track roll control: bank toward the gate line (body-right error).
            e_cross = dx * math.sin(yaw) - dy * math.cos(yaw)
            tgt_roll = float(np.clip(args.klat * e_cross, -0.12, 0.12))
            # Track-data gate z appears sign-flipped vs odometry NED (course climbs).
            # zoff lifts the aim point so we clear the bottom border of the opening.
            tgt_z = (-g[2] if args.flipz else g[2]) + args.zoff

        # Attitude-angle P -> rate commands, with measured sign conventions.
        roll_cmd = float(
            np.clip(s_roll * K_ATT * (tgt_roll - roll), -RATE_CLIP, RATE_CLIP)
        )
        pitch_cmd = float(
            np.clip(s_pitch * K_ATT * (tgt_pitch - pitch), -RATE_CLIP, RATE_CLIP)
        )
        yaw_cmd = float(np.clip(s_yaw * K_YAW * yaw_err, -YAW_CLIP, YAW_CLIP))
        # Vision mode: bound vertical authority hard. A collision-opened
        # covariance lets one landmark yank z metres from target; unclamped
        # PD then saturates thrust (0.5 = ~18 m/s^2, measured: rockets to the
        # ceiling). Clamp the error AND the thrust range (~+/-3 m/s^2 max).
        if args.mode == "vision":
            z_err = float(np.clip(z - tgt_z, -3.0, 3.0))
            vz_c = float(np.clip(vz, -4.0, 4.0))
            # Floor 0.17 (was 0.20): the gate opening is a 1.5 m band and
            # crossings carry ~1.2 m/s climb -- the brake needs enough
            # down-authority to arrest it inside the band (2026-07-05).
            thrust = float(np.clip(HOVER_T + KP_Z * z_err + KD_Z * vz_c, 0.17, 0.36))
            if takeoff_boost:
                thrust = max(thrust, 0.32)  # complete the takeoff climb in time
        else:
            thrust = float(np.clip(HOVER_T + KP_Z * (z - tgt_z) + KD_Z * vz, 0.18, 0.5))
        # Lost (scanning >1.5s with no confirmed gate): the z estimate can't
        # be trusted, so bleed REAL altitude gently -- gates live near the
        # floor; this brings a ceiling-wanderer back into vision range.
        if args.mode == "vision" and scan_since is not None:
            if time.time() - scan_since > 1.5:
                thrust = HOVER_T - 0.012
        sim.send_attitude_rates(roll_cmd, pitch_cmd, yaw_cmd, thrust)

        # Safety.
        gb_z = (spec.quat_to_R(snap.quat).T @ np.array([0.0, 0, 1.0]))[2]
        if gb_z < 0.0:
            reason = "ABORT flipped"
            break
        # In est mode z is an ESTIMATE with no baro backup -- a tight bound
        # aborts recoverable runs on estimator drift alone.
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
    print(
        f"[f2] === DONE {reason} final={np.round(sim.snapshot().pos_ned, 1)} "
        f"active={sim.data.get('active_gate_index')} ===",
        flush=True,
    )
    # Drain the flight log + print the shadow accuracy report BEFORE
    # os._exit, which skips atexit/finally (abort paths land here too).
    sim.finish_flight(
        extra={
            "gates_passed_vision": getattr(guide, "n_passed", None),
            "active_gate_index_sim": sim.data.get("active_gate_index"),
            "reason": reason,
        }
    )
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
