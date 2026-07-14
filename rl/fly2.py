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
import csv
import json
import math
import os
import sys
import time

import numpy as np

from rl import spec
from rl.fly2_course import (
    EST_SIGNS,
    HOVER_T,
    Fly2Config,
    GateCarrot,
    RateCmdShaper,
    detect_climb_course,
    rates_from_attitude_targets,
    rpy,
    wrap,
)
from rl.sim_interface import GATE_MAP_PATH, SimInterface
from simulator import display
from simulator.preflight import wait_for_race_go
from simulator.tilt_filter import TiltFilter
from simulator.vision_nav import VisionGuidance, VisualServo

HZ = 90.0  # spec VADR-TS-003 4.4: command rate must stay < 100 Hz
TAKEOFF_S = 4.0
TAKEOFF_LEVEL_RAD = 0.18  # ~10° — leave takeoff once closer than this to level
TAKEOFF_THRUST = 0.30  # soft lift once upright; 0.35+ on ramp pitch runaway
TAKEOFF_IDLE_THRUST = 0.22  # below hover while pitched — no +x accel on ramp
HOVER_Z_NED = -3.0
TAKEOFF_KD = 0.12  # gyro damping while leveling (pure-P was under-damped)


def _alt_src(sim: SimInterface, est_mode: bool) -> str:
    if sim.data.get("odometry") is not None:
        return "odo"
    if sim.data.get("attitude") is not None:
        return "att" if sim.data.get("has_position") else "att+ekf_pos"
    if est_mode:
        return "ekf"
    return "?"


def _wrap_att(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def _control_rpy(
    snap, sim: SimInterface, tilt: TiltFilter, est_mode: bool, thrusting: bool
):
    """Roll/pitch for the rate law — never chase EKF euler under thrust.

    Returns (roll, pitch, yaw, att_src).
    """
    imu = snap.imu or sim.data.get("imu")
    att = sim.data.get("attitude")
    if sim.data.get("odometry") is not None and snap.quat is not None:
        r, p, y = rpy(snap.quat)
        return r, p, y, "odo"
    if att is not None:
        return (
            float(att["roll"]),
            float(att["pitch"]),
            float(att.get("yaw", 0.0)),
            "mav_attitude",
        )
    if imu is not None:
        # Gyro-only while thrusting: accel measures thrust, not gravity.
        tilt.update(imu, allow_accel=not thrusting, boost=False)
        yaw = float(sim.data.get("yaw_rad") or 0.0)
        return _wrap_att(tilt.roll), _wrap_att(tilt.pitch), yaw, "tilt"
    if snap.quat is not None:
        r, p, y = rpy(snap.quat)
        return r, p, y, "snap_quat_ekf_fallback"
    return 0.0, 0.0, 0.0, "none"


def _rate_signs(att_src: str, att_signs):
    """Command signs for the active attitude source.

    signs.json pitch=+1 matches ODOMETRY. With IMU tilt (VQ2 / no odo), live
    logs 2026-07-13 showed +pitch_cmd @ signs=(−1,+1,−1) with gy≈−0.76 and
    pitch −18°→−91° (positive feedback / ABORT flipped). Flip pitch for tilt
    (and EKF-quat fallback) only — leave odometry/MAV attitude unchanged.
    """
    from rl.fly2_course import _default_signs

    s = att_signs if att_signs is not None else _default_signs()
    if att_src in ("tilt", "snap_quat_ekf_fallback"):
        return (float(s[0]), float(-s[1]), float(s[2]))
    return (float(s[0]), float(s[1]), float(s[2]))


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
        default=0.0,
        help="altitude trim vs gate OPENING CENTRE (negative = fly higher, NED); "
        "centre = gate base + h/2, so 0 = dead centre",
    )
    ap.add_argument("--klat", type=float, default=0.04, help="cross-track roll gain")
    ap.add_argument(
        "--kd",
        type=float,
        default=None,
        help="override course/vision body-rate damping k_d (live A/B; default: "
        "measured flightlab/calibration.json value, else 0)",
    )
    ap.add_argument(
        "--wait",
        action="store_true",
        help="optional: pause for ENTER before race-GO wait (debug sync)",
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
        if not args.flipz and detect_climb_course(gate_map):
            cfg.flipz = True
            print("[f2] climb course detected — flipz=True", flush=True)
    guide = None  # constructed after est_mode is known (speed differs)
    sim = SimInterface(use_estimator=args.est)
    # Quiet per-frame GATE spam so race-GO / arm lines stay readable.
    sim.data["_quiet_vision"] = True
    if not sim.wait_for_telemetry():
        print("[f2] no telemetry", flush=True)
        os._exit(1)
    if args.reset:
        sim.reset_sim()
        time.sleep(3)
        # The teleport invalidates the gyro-integrated attitude: re-init.
        sim.estimator.reset()
        time.sleep(1.5)
    # Race GO alone gates launch (countdown → 0). Optional --wait for manual sync.
    if args.wait and args.mode in ("course", "vision"):
        print(
            "[f2] --wait: press ENTER once, then wait for race GO...",
            flush=True,
        )
        try:
            input("[f2] press ENTER to continue to race-GO wait...")
        except EOFError:
            pass
    print(
        "[f2] waiting for race GO — click Race in FlightSim "
        "(arms + flies when on-screen timer hits 0)...",
        flush=True,
    )
    if not wait_for_race_go(sim.data, timeout_s=90.0):
        print(
            "[f2] race GO timeout — in FlightSim: enter session, click Race, "
            "let countdown reach 0",
            flush=True,
        )
        os._exit(1)
    sim.arm()
    if not sim.ensure_armed(timeout_s=8.0):
        print(
            "[f2] armed heartbeat not seen — continuing (sending arm+setpoints anyway)",
            flush=True,
        )
    s0 = sim.snapshot()
    if not s0.has_pose():
        print("[f2] no pose after arm", flush=True)
        os._exit(1)
    hold_z = s0.pos_ned[2]
    hold_yaw = rpy(s0.quat)[2]
    takeoff_z = min(hold_z, HOVER_Z_NED)
    # EKF attitude uses odometry sign convention; EST_SIGNS only for --est experiments.
    est_mode = args.est or sim.data.get("odometry") is None
    att_signs = EST_SIGNS if args.est else None
    kd_course = args.kd if args.kd is not None else (TAKEOFF_KD if est_mode else None)
    carrot = GateCarrot()
    shaper = RateCmdShaper()
    if args.mode == "vision":
        # Estimator regime: body-frame visual servoing -- world-position
        # drift cannot create phantom targets (measured failure of the
        # map-based guidance without odometry). Training/odometry keeps the
        # original world-map guidance.
        guide = VisualServo() if est_mode else VisionGuidance()
    print(
        f"[f2] mode={args.mode} hold_z={hold_z:.1f} takeoff_z={takeoff_z:.1f}"
        f" hover_t={HOVER_T} est_mode={est_mode} armed={s0.armed}",
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

    # Per-run course telemetry at loop rate — post-analyze smoothness with
    # flightlab.metrics.analyze_jitter (roll/pitch std, rate_cmd_std, PSD).
    course_log, course_wr = None, None
    if args.mode == "course":
        os.makedirs(os.path.join("rl", "data"), exist_ok=True)
        course_path = os.path.join(
            "rl", "data", time.strftime("course_log_%Y%m%d_%H%M%S.csv")
        )
        course_log = open(course_path, "w", newline="")
        course_wr = csv.writer(course_log)
        course_wr.writerow(
            "t stage roll_deg pitch_deg yaw_deg cmd_rr cmd_pr cmd_yr thrust "
            "z active att_src armed".split()
        )
        print(f"[f2] course log -> {course_path}", flush=True)

    display.start()  # live vision window (what the drone's camera sees)

    tilt = TiltFilter()
    # Seed tilt from ground accel before lift (ramp pitch is real).
    imu0 = sim.data.get("imu")
    if imu0 is not None:
        tilt.update(imu0, allow_accel=True, boost=True)
        tilt.sync_to_accel()

    t0 = time.time()
    last_log = 0.0
    last_active = -1
    last_shown_tag = None
    last_pose_fid = None
    last_col = sim.data.get("last_collision")  # ignore stale pre-run hits
    last_col_t = -1e9
    scan_since = None
    last_arm_t = 0.0
    takeoff_done = args.mode == "hover"
    reason = "timeout"
    while time.time() - t0 < args.seconds:
        if not sim.data.get("armed") and time.time() - last_arm_t >= 1.0:
            sim.arm()
            last_arm_t = time.time()
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
        z, vz = p[2], v[2]
        in_takeoff = (not takeoff_done) and args.mode in ("course", "vision")
        thrusting = in_takeoff or args.mode != "hover"
        roll, pitch, yaw, att_src = _control_rpy(
            snap, sim, tilt, est_mode, thrusting=thrusting
        )
        signs = _rate_signs(att_src, att_signs)
        gyro = snap.ang_vel or (0.0, 0.0, 0.0)
        if snap.imu is not None and (gyro is None or gyro == (0.0, 0.0, 0.0)):
            gyro = (
                snap.imu.get("gx", 0.0),
                snap.imu.get("gy", 0.0),
                snap.imu.get("gz", 0.0),
            )

        # Face the active/first gate (spawn yaw=0, gates at −x → yaw_err≈±180).
        gate_i = int(sim.data.get("active_gate_index", 0) or 0)
        if gate_map and 0 <= gate_i < len(gate_map):
            gp = gate_map[gate_i]["pos"]
            bearing = math.atan2(gp[1] - p[1], gp[0] - p[0])
            course_yaw_err = wrap(bearing - yaw)
        else:
            bearing = yaw
            course_yaw_err = 0.0

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
                signs=signs,
                k_d=TAKEOFF_KD,
                gyro=gyro,
            )
        elif in_takeoff:
            # Level + yaw to face gates first. Idle thr while pitched so ramp
            # nose-down does not accelerate +x (away from course) into a flip.
            level = abs(roll) < TAKEOFF_LEVEL_RAD and abs(pitch) < TAKEOFF_LEVEL_RAD
            tgt_z = takeoff_z if level else hold_z
            roll_cmd, pitch_cmd, yaw_cmd, thrust = rates_from_attitude_targets(
                roll,
                pitch,
                z,
                vz,
                0.0,
                0.0,
                course_yaw_err,
                tgt_z,
                signs=signs,
                k_d=TAKEOFF_KD,
                gyro=gyro,
            )
            if level:
                thrust = float(
                    np.clip(
                        min(thrust, TAKEOFF_THRUST), TAKEOFF_IDLE_THRUST, TAKEOFF_THRUST
                    )
                )
            else:
                thrust = TAKEOFF_IDLE_THRUST
            vstatus = "TAKEOFF"
            if level and time.time() - t0 > 1.0:
                takeoff_done = True
            elif time.time() - t0 >= TAKEOFF_S:
                takeoff_done = True
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
                signs=signs,
                k_d=kd_course,
                gyro=gyro,
            )
            # Vision mode: bound vertical authority hard.
            z_err = float(np.clip(z - cmd.tgt_z, -3.0, 3.0))
            vz_c = float(np.clip(vz, -4.0, 4.0))
            thrust = float(np.clip(HOVER_T + 0.025 * z_err + 0.030 * vz_c, 0.20, 0.36))
            if scan_since is not None and time.time() - scan_since > 1.5:
                thrust = HOVER_T - 0.012
        else:
            active = gate_i
            if active != last_active:
                print(
                    f"[f2] [{time.time() - t0:4.1f}s] ACTIVE GATE -> {active}",
                    flush=True,
                )
                last_active = active
            if active >= n:
                reason = "COURSE COMPLETE"
                break
            speed = float(np.linalg.norm(v[:2]))
            # Carrot aim: slews across gate transitions instead of stepping
            # bearing / cross-track / tgt_z the tick active_gate_index advances.
            ax, ay, tgt_z = carrot.update(active, gate_map, cfg)
            dx, dy = ax - p[0], ay - p[1]
            aim_yaw_err = wrap(math.atan2(dy, dx) - yaw)
            e_cross = dx * math.sin(yaw) - dy * math.cos(yaw)
            align = max(0.0, 1.0 - abs(aim_yaw_err) / 0.4)
            dist = math.hypot(dx, dy)
            v_des = cfg.speed * align * min(1.0, 0.3 + dist / 10.0)
            lean = float(np.clip(0.05 * (v_des - speed), -0.05, cfg.lean))
            roll_cmd, pitch_cmd, yaw_cmd, thrust = rates_from_attitude_targets(
                roll,
                pitch,
                z,
                vz,
                float(np.clip(cfg.klat * e_cross, -0.12, 0.12)),
                -lean,
                aim_yaw_err,
                tgt_z,
                signs=signs,
                k_d=kd_course,
                gyro=gyro,
            )

        # Shape every outgoing command (covers the takeoff -> course seam and
        # mode branches alike): low-pass + slew on rates, slew on thrust.
        roll_cmd, pitch_cmd, yaw_cmd, thrust = shaper.apply(
            roll_cmd, pitch_cmd, yaw_cmd, thrust
        )
        sim.send_attitude_rates(roll_cmd, pitch_cmd, yaw_cmd, thrust)

        if course_wr is not None:
            course_wr.writerow(
                [
                    round(time.time() - t0, 3),
                    "takeoff" if in_takeoff else "course",
                    round(math.degrees(roll), 2),
                    round(math.degrees(pitch), 2),
                    round(math.degrees(yaw), 2),
                    round(roll_cmd, 4),
                    round(pitch_cmd, 4),
                    round(yaw_cmd, 4),
                    round(thrust, 4),
                    round(z, 2),
                    gate_i,
                    att_src,
                    snap.armed,
                ]
            )

        # Flip detect from the attitude we actually control with.
        if math.cos(roll) * math.cos(pitch) < 0.0:
            reason = "ABORT flipped"
            break
        z_abort = 60 if est_mode else 30
        if z < hold_z - z_abort or z > hold_z + z_abort:
            reason = "ABORT altitude"
            break

        now = time.time() - t0
        if now - last_log >= 0.5:
            print(
                f"[f2] [{now:4.1f}s] armed={snap.armed} alt={_alt_src(sim, est_mode)} "
                f"src={att_src} "
                f"rpy=({math.degrees(roll):+4.0f},"
                f"{math.degrees(pitch):+4.0f},{math.degrees(yaw):+4.0f}) "
                f"z={z:+5.1f} v=({v[0]:+4.1f},{v[1]:+4.1f},{v[2]:+4.1f}) "
                f"thr={thrust:.2f} pcmd={pitch_cmd:+.2f} ycmd={yaw_cmd:+.2f}"
                f"{('  ' + vstatus) if vstatus else ''}",
                flush=True,
            )
            last_log = now
        time.sleep(1 / HZ)

    sim.send_attitude_rates(0, 0, 0, HOVER_T)
    display.close()
    if course_log is not None:
        course_log.close()
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
