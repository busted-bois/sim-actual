"""Vision-fed flight — the reference architecture: CNN vision + tuned control.

Uses fly2's MEASURED-dynamics controller verbatim (the proven, stable flight),
but the gate target comes from the YOLO-pose -> PnP -> world-tracker pipeline
(Modules 3-4c) instead of the raw broadcast gate map. Vision refines the gate
POSITION; sequencing still uses the sim's active_gate_index.

Why a thread: YOLO inference on CPU is ~100 ms. Running it inline would stall
the 150 Hz control loop and make the drone shake (exactly what the raw-RL
deploy did). So YOLO runs in its own thread and the control loop only reads the
latest smoothed track — control never blocks on vision.

This path uses gate POSITIONS only (bearing + cross-track + altitude), never the
gate normal, so it is immune to the gate-normal sign question.

    uv run -m rl.fly_vision --seconds 90
"""

from __future__ import annotations

import argparse
import glob
import math
import os
import sys
import threading
import time

import cv2
import numpy as np

from rl import spec
from rl.fly2 import (
    HOVER_T,
    K_ATT,
    K_YAW,
    KD_Z,
    KP_Z,
    RATE_CLIP,
    SIGN_PITCH,
    SIGN_ROLL,
    SIGN_YAW,
    YAW_CLIP,
    rpy,
    wrap,
)
from rl.dataset import (
    POSE_OUT_DIR,
    _yolo_pose_line,
    project_gate_keypoints,
    write_pose_data_yaml,
)
from rl.gate_tracker import GateTracker
from rl.gatepose import visible_mask
from rl.sim_interface import GATE_MAP_PATH, SimInterface, load_gate_map

HZ = 150.0
ACQUIRE_CREEP = 0.025  # vision-only: gentle forward lean while searching for a gate
PUNCH_S = 1.5  # vision-only: seconds to fly straight forward after passing a gate
PUNCH_LEAN = 0.06  # forward lean during the punch-through (clears the opening)
_CORNER_NAMES = ("TL", "TR", "BR", "BL")


def _draw_det(ann, det, pose):
    """Draw one YOLO detection on the live-view frame: bbox, 4 labeled corners
    (green = used by PnP, red = rejected), confidence + PnP range."""
    vm = visible_mask(det)
    x0, y0, x1, y1 = det.bbox.astype(int)
    cv2.rectangle(ann, (x0, y0), (x1, y1), (255, 255, 0), 1)
    pts = det.corners.astype(int)
    cv2.polylines(ann, [pts.reshape(-1, 1, 2)], True, (0, 200, 0), 1)
    for i, (px, py) in enumerate(pts):
        col = (0, 255, 0) if vm[i] else (0, 0, 255)
        cv2.circle(ann, (int(px), int(py)), 4, col, -1)
        cv2.putText(
            ann,
            _CORNER_NAMES[i],
            (int(px) + 4, int(py) - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            col,
            1,
        )
    label = f"{det.det_conf:.2f}"
    if pose is not None:
        label += f" {pose['range_m']:.1f}m"
    cv2.putText(
        ann,
        label,
        (x0, max(12, y0 - 4)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (0, 255, 255),
        1,
    )


class VisionNav:
    """Pick the flight target from VISION TRACKS ONLY — no broadcast gate map.

    Acquires the nearest fresh (un-passed) gate track, follows its refined
    position, and HOLDS it through close-range detection dropout until the drone
    passes within `pass_r`; then advances to the next nearest un-passed gate.
    Sequences the whole course purely from what vision saw (at the start the
    drone sees all gates at once, so the tracker has every gate's position).
    """

    def __init__(self, pass_r=1.8, dedup=3.0, refresh=2.5):
        self.target = None
        self.passed = []
        self.n_passed = 0
        self.just_passed = False
        self.pass_r = pass_r
        self.dedup = dedup
        self.refresh = refresh

    def update(self, tracks, p):
        self.just_passed = False
        if self.target is not None:
            near = [t for t in tracks if np.linalg.norm(t - self.target) < self.refresh]
            if near:  # follow the refined estimate while the gate is still seen
                self.target = min(near, key=lambda t: np.linalg.norm(t - self.target))
            if np.linalg.norm(self.target - p) < self.pass_r:  # essentially at the gate
                self.passed.append(self.target)
                self.n_passed += 1
                self.target = None
                self.just_passed = True
        if self.target is None:  # acquire nearest fresh gate ahead
            fresh = [
                t
                for t in tracks
                if np.linalg.norm(t - p) > self.pass_r
                and all(np.linalg.norm(t - q) > self.dedup for q in self.passed)
            ]
            if fresh:
                self.target = min(fresh, key=lambda t: np.linalg.norm(t - p))
        return self.target


class VisionThread(threading.Thread):
    """Continuously run YOLO->PnP on the newest frame and feed the tracker.

    Decoupled from control: updates at whatever rate YOLO sustains (~5-10 Hz on
    CPU) while the control loop runs at 150 Hz off the latest smoothed tracks.
    """

    def __init__(self, sim: SimInterface, tracker: GateTracker, view: bool = False):
        super().__init__(daemon=True)
        self.sim = sim
        self.tracker = tracker
        self.view = view
        self.annotated = None  # latest frame with detections drawn (for the window)
        self.running = True
        self.ready = False
        self.last_fid = -1
        self.n_raw = 0
        self.n_seen = 0
        self.infer_ms = 0.0
        self.best = None  # (nx, ny, r_frac) of the locked gate, image-space
        self.best_t = 0.0
        self._best_xy = None  # last locked gate pixel center (for continuity)

    def run(self):
        from rl import perception as P

        try:
            perc = P.GatePerception()
        except Exception as e:  # missing weights / ultralytics
            print(f"[fv] vision OFF — {e}", flush=True)
            return
        self.ready = True
        print("[fv] vision thread up (YOLO-pose -> PnP -> tracker)", flush=True)
        while self.running:
            frame = self.sim.data.get("frame")
            odo = self.sim.data.get("odometry")
            if not frame or odo is None or frame["frame_id"] == self.last_fid:
                time.sleep(0.005)
                continue
            self.last_fid = frame["frame_id"]
            dpos = np.array([odo["x"], odo["y"], odo["z"]])
            dq = np.array([odo["qw"], odo["qx"], odo["qy"], odo["qz"]])
            t0 = time.monotonic()
            dets = perc.infer.detect(frame["img"])  # raw YOLO detections
            ann = frame["img"].copy() if self.view else None
            poses = []
            for d in dets:
                g = P.gate_world_pose(d, dpos, dq)
                if g is not None:
                    poses.append(g)
                if ann is not None:
                    _draw_det(ann, d, g)
            self.infer_ms = (time.monotonic() - t0) * 1e3
            if ann is not None:
                self.annotated = ann
            self.tracker.update([g["gate_pos_world"] for g in poses], time.monotonic())
            self.n_raw = len(dets)
            self.n_seen = len(poses)
            # Locked gate's image-space center (for visual servoing). LOCK ONTO
            # ONE gate via continuity (nearest to the previous lock), not the
            # largest box each frame — that flickers between the stacked gates of
            # a descending course and makes the yaw thrash.
            if dets:
                h, w = frame["img"].shape[:2]
                cands = []
                for dd in dets:
                    a = (dd.bbox[2] - dd.bbox[0]) * (dd.bbox[3] - dd.bbox[1])
                    cands.append((
                        (dd.bbox[0] + dd.bbox[2]) / 2, (dd.bbox[1] + dd.bbox[3]) / 2, a
                    ))
                if self._best_xy is not None:
                    cx, cy, a = min(
                        cands,
                        key=lambda c: (c[0] - self._best_xy[0]) ** 2
                        + (c[1] - self._best_xy[1]) ** 2,
                    )
                    # Lock jumped far (gate gone/passed) -> re-lock to the largest.
                    if (cx - self._best_xy[0]) ** 2 + (cy - self._best_xy[1]) ** 2 > 160**2:
                        cx, cy, a = max(cands, key=lambda c: c[2])
                else:
                    cx, cy, a = max(cands, key=lambda c: c[2])  # seed with closest
                self._best_xy = (cx, cy)
                self.best = ((cx - w / 2) / (w / 2), (cy - h / 2) / (h / 2),
                             float(a) / (w * h))
                self.best_t = time.monotonic()
            else:
                self.best = None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=95.0)
    ap.add_argument("--speed", type=float, default=2.8)
    ap.add_argument("--lean", type=float, default=0.12, help="max forward lean (rad)")
    ap.add_argument(
        "--zoff",
        type=float,
        default=-1.0,
        help="altitude offset vs gate (NED, - = higher)",
    )
    ap.add_argument("--klat", type=float, default=0.04, help="cross-track roll gain")
    ap.add_argument("--no-wait", dest="wait", action="store_false")
    ap.add_argument("--reset", action="store_true")
    ap.add_argument(
        "--vision-only",
        action="store_true",
        help="fly ONLY on vision tracks — no broadcast gate map used for targets",
    )
    ap.add_argument(
        "--gates", type=int, default=6, help="gate count (vision-only finish)"
    )
    ap.add_argument(
        "--no-view",
        dest="view",
        action="store_false",
        help="disable the live vision window (on by default)",
    )
    ap.add_argument(
        "--record",
        type=int,
        default=0,
        help="also save N frames + auto-labels to the pose dataset while flying "
        "(captures the real close-range trajectory; appends to existing data)",
    )
    args = ap.parse_args()

    gate_map = load_gate_map(GATE_MAP_PATH)
    n = len(gate_map)
    sim = SimInterface()
    if not sim.wait_for_telemetry():
        print("[fv] no telemetry", flush=True)
        os._exit(1)

    tracker = GateTracker()
    vision = VisionThread(sim, tracker, view=args.view)
    vision.start()

    if args.reset:
        sim.reset_sim()
        time.sleep(3)
    if args.wait:
        try:
            input("[fv] READY -- press ENTER the moment the countdown hits 0...")
        except EOFError:
            pass
    sim.arm()
    time.sleep(0.2)
    s0 = sim.snapshot()
    hold_z = s0.pos_ned[2]
    nav = VisionNav() if args.vision_only else None
    mode = "VISION-ONLY (no broadcast map)" if args.vision_only else "broadcast+vision"
    print(f"[fv] flying [{mode}]; hold_z={hold_z:.1f} hover_t={HOVER_T}", flush=True)

    # Optional data recording: save frames + auto-labels along the real flown
    # trajectory (close-range + all gates the controller reaches), appended to
    # the existing pose dataset. Fixes the far-only bias of the HSV-pilot set.
    rec_on = args.record > 0
    rec_saved, rec_fid = 0, -1
    rec_idx = 0
    if rec_on:
        for sp in ("train", "val"):
            os.makedirs(os.path.join(POSE_OUT_DIR, "images", sp), exist_ok=True)
            os.makedirs(os.path.join(POSE_OUT_DIR, "labels", sp), exist_ok=True)
        existing = glob.glob(os.path.join(POSE_OUT_DIR, "images", "*", "*.png"))
        rec_idx = (
            max(
                (int(os.path.splitext(os.path.basename(p))[0]) for p in existing),
                default=-1,
            )
            + 1
        )
        print(
            f"[fv] recording {args.record} frames -> {POSE_OUT_DIR} "
            f"(start idx {rec_idx})",
            flush=True,
        )

    t0 = time.time()
    last_log = 0.0
    punch_until = 0.0
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

        # Record frame + auto-label along the flown trajectory.
        if rec_on and rec_saved < args.record and snap.frame is not None:
            fr = sim.data.get("frame")
            if fr and fr["frame_id"] != rec_fid:
                rec_fid = fr["frame_id"]
                labels = project_gate_keypoints(
                    snap.frame.shape[:2], p, np.asarray(snap.quat, float), gate_map
                )
                if labels or (rec_saved % 10) < 3:  # keep ~30% negatives
                    sp = "val" if (rec_idx % 6 == 0) else "train"
                    cv2.imwrite(
                        os.path.join(POSE_OUT_DIR, "images", sp, f"{rec_idx:06d}.png"),
                        snap.frame,
                    )
                    with open(
                        os.path.join(POSE_OUT_DIR, "labels", sp, f"{rec_idx:06d}.txt"),
                        "w",
                    ) as f:
                        f.write("\n".join(_yolo_pose_line(b) for b in labels))
                    rec_idx += 1
                    rec_saved += 1
                    if rec_saved % 50 == 0:
                        print(f"[fv] recorded {rec_saved}/{args.record}", flush=True)

        if args.vision_only:
            # Target comes ONLY from vision tracks — gate_map is never read here.
            tgt = nav.update(tracker.tracks(min_n=2), p)
            if nav.just_passed:
                punch_until = time.time() + PUNCH_S
            if nav.n_passed >= args.gates:
                reason = "COURSE COMPLETE (vision-only)"
                break
            gate_lbl = f"g{nav.n_passed}"
            if time.time() < punch_until:
                # Just passed a gate — punch straight forward to clear the opening
                # before turning toward the next gate.
                src = "PUNCH"
                tgt_roll = 0.0
                tgt_pitch = -PUNCH_LEAN
                yaw_err = 0.0
                tgt_z = hold_z
                dist = 0.0
            elif tgt is None:
                # No gate acquired yet — creep gently forward (keep heading) to
                # bring a gate into reliable detection range, holding altitude.
                src = "ACQUIRE"
                tgt_roll = 0.0
                tgt_pitch = -ACQUIRE_CREEP
                yaw_err = 0.0
                tgt_z = hold_z
                dist = 0.0
            else:
                src = "VIS-ONLY"
                g = tgt
                dx, dy = g[0] - p[0], g[1] - p[1]
                dist = math.hypot(dx, dy)
                yaw_err = wrap(math.atan2(dy, dx) - yaw)
                speed = float(np.linalg.norm(v[:2]))
                align = max(0.0, 1.0 - abs(yaw_err) / 0.4)
                v_des = args.speed * align * min(1.0, 0.3 + dist / 10.0)
                lean = float(np.clip(0.05 * (v_des - speed), -0.05, args.lean))
                tgt_pitch = -lean
                e_cross = dx * math.sin(yaw) - dy * math.cos(yaw)
                tgt_roll = float(np.clip(args.klat * e_cross, -0.12, 0.12))
                tgt_z = g[2] + args.zoff
        else:
            # Broadcast+vision: target is the broadcast gate, refined by a matched
            # vision track when one exists (within tracker.match_dist).
            active = int(sim.data.get("active_gate_index", 0) or 0)
            if active >= n:
                reason = "COURSE COMPLETE"
                break
            eff = tracker.corrected_map(gate_map)
            g = np.asarray(eff[active]["pos"], float)
            used_vision = not np.allclose(g, np.asarray(gate_map[active]["pos"], float))
            gate_lbl = f"g{active}"
            src = "VISION-corr" if used_vision else "bcast"
            dx, dy = g[0] - p[0], g[1] - p[1]
            dist = math.hypot(dx, dy)
            yaw_err = wrap(math.atan2(dy, dx) - yaw)
            speed = float(np.linalg.norm(v[:2]))
            align = max(0.0, 1.0 - abs(yaw_err) / 0.4)
            v_des = args.speed * align * min(1.0, 0.3 + dist / 10.0)
            lean = float(np.clip(0.05 * (v_des - speed), -0.05, args.lean))
            tgt_pitch = -lean
            e_cross = dx * math.sin(yaw) - dy * math.cos(yaw)
            tgt_roll = float(np.clip(args.klat * e_cross, -0.12, 0.12))
            tgt_z = g[2] + args.zoff

        # fly2's measured-dynamics attitude-rate law (unchanged, stable).
        roll_cmd = float(
            np.clip(SIGN_ROLL * K_ATT * (tgt_roll - roll), -RATE_CLIP, RATE_CLIP)
        )
        pitch_cmd = float(
            np.clip(SIGN_PITCH * K_ATT * (tgt_pitch - pitch), -RATE_CLIP, RATE_CLIP)
        )
        yaw_cmd = float(np.clip(SIGN_YAW * K_YAW * yaw_err, -YAW_CLIP, YAW_CLIP))
        thrust = float(np.clip(HOVER_T + KP_Z * (z - tgt_z) + KD_Z * vz, 0.18, 0.5))
        sim.send_attitude_rates(roll_cmd, pitch_cmd, yaw_cmd, thrust)

        gb_z = (spec.quat_to_R(snap.quat).T @ np.array([0.0, 0, 1.0]))[2]
        if gb_z < 0.0:
            reason = "ABORT flipped"
            break
        if z < hold_z - 30 or z > hold_z + 30:
            reason = "ABORT altitude"
            break

        n_tracks = len(tracker.tracks(min_n=2))
        now = time.time() - t0
        if now - last_log >= 0.5:
            print(
                f"[fv] [{now:4.1f}s] {gate_lbl} {src} raw={vision.n_raw} "
                f"det={vision.n_seen} tracks={n_tracks} yolo={vision.infer_ms:.0f}ms "
                f"rpy=({math.degrees(roll):+4.0f},{math.degrees(pitch):+4.0f},"
                f"{math.degrees(yaw):+4.0f}) z={z:+5.1f} dist={dist:4.1f} thr={thrust:.2f}",
                flush=True,
            )
            last_log = now

        if args.view and vision.annotated is not None:
            hud = vision.annotated
            cv2.putText(
                hud,
                f"{gate_lbl} {src} raw={vision.n_raw} det={vision.n_seen} "
                f"tracks={n_tracks} dist={dist:.1f}m",
                (6, 18),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
            )
            cv2.imshow("vision", hud)
            cv2.waitKey(1)
        time.sleep(1 / HZ)

    vision.running = False
    if args.view:
        cv2.destroyAllWindows()
    if rec_on:
        write_pose_data_yaml(POSE_OUT_DIR)
        print(f"[fv] recorded {rec_saved} frames -> {POSE_OUT_DIR}", flush=True)
    sim.send_attitude_rates(0, 0, 0, HOVER_T)
    print(
        f"[fv] === DONE {reason} final={np.round(sim.snapshot().pos_ned, 1)} "
        f"active={sim.data.get('active_gate_index')} ===",
        flush=True,
    )
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
