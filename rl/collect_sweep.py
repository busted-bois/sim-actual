"""Diversity-sweep data collector — balanced coverage of ALL gates.

Flies a planned path that approaches EACH gate from its approach side at several
ranges (12/8/5/2 m and just-past) with lateral variation, recording frames +
range-gated auto-labels the whole time. Uses the broadcast gate positions only
to (a) plan the camera path and (b) auto-label (standard supervised labeling —
the trained detector uses pixels only at inference).

This fixes the far-/close-bias of the earlier sets: every gate gets seen at the
action range (3-15 m) from varied angles. Appends to the existing pose dataset.

    uv run -m rl.collect_sweep                 # fly the sweep, record
    uv run -m rl.collect_sweep --selftest      # offline: trajectory-gen sanity
"""

from __future__ import annotations

import argparse
import glob
import math
import os
import time

import cv2
import numpy as np

from rl.dataset import (
    POSE_OUT_DIR,
    _yolo_pose_line,
    project_gate_keypoints,
    write_pose_data_yaml,
)
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
from rl.sim_interface import GATE_MAP_PATH, SimInterface, load_gate_map

HZ = 150.0
SWEEP_RANGES = (15.0, 11.0, 8.0, 5.0, 2.0, -1.5)  # m along approach; <0 = past gate
SWEEP_LATERAL = (0.0, 3.0, -3.0, 1.5)  # m sideways, cycled for angle diversity
SWEEP_ALT = (0.0, -1.5, 1.0)  # m altitude offset (NED), cycled for elevation diversity
ARRIVE_R = 2.5  # m: waypoint reached (loose — fly2 can't hold a tight point)
WP_TIMEOUT_S = 5.0  # advance to next waypoint after this even if not reached
# (stops the sweep getting stuck circling an unreachable lateral/alt waypoint)
LABEL_MAX_RANGE = 28.0  # m: label gates out to here (covers the ~22 m start); the
# perception span-gate handles unreliable far PnP, so we DON'T starve detection.
ZOFF = -1.0  # aim slightly above gate center (NED)


def _horiz_unit(v):
    v = np.array([v[0], v[1], 0.0], dtype=np.float64)
    n = np.linalg.norm(v)
    return v / n if n > 1e-6 else np.array([1.0, 0.0, 0.0])


def plan_sweep(gate_map: list, start_pos) -> list[np.ndarray]:
    """Flat waypoint list: approach each gate from its approach side (toward the
    previous gate / start) at SWEEP_RANGES with lateral variation, ending just
    past the gate (fly through). Yaw follows motion, so the gate stays ahead in
    view during each approach."""
    wps: list[np.ndarray] = []
    prev = np.asarray(start_pos, dtype=np.float64)
    for gi, g in enumerate(gate_map):
        gc = np.asarray(g["pos"], dtype=np.float64)
        back = _horiz_unit(prev - gc)  # unit vector toward the approach side
        right = np.array([-back[1], back[0], 0.0])  # horizontal perpendicular
        for k, r in enumerate(SWEEP_RANGES):
            lat = SWEEP_LATERAL[(gi + k) % len(SWEEP_LATERAL)] if r > 0 else 0.0
            alt = SWEEP_ALT[(gi + k) % len(SWEEP_ALT)] if r > 0 else 0.0
            wps.append(gc + back * r + right * lat + np.array([0.0, 0.0, alt]))
        prev = gc
    return wps


def collect(out_dir: str = POSE_OUT_DIR, seconds: float = 180.0, wait: bool = True):
    from simulator.controller import Controller  # noqa: F401 (keeps sim import local)

    gate_map = load_gate_map(GATE_MAP_PATH)
    sim = SimInterface()
    if not sim.wait_for_telemetry():
        print("[sweep] no telemetry — is the sim running?", flush=True)
        return
    for sp in ("train", "val"):
        os.makedirs(os.path.join(out_dir, "images", sp), exist_ok=True)
        os.makedirs(os.path.join(out_dir, "labels", sp), exist_ok=True)
    existing = glob.glob(os.path.join(out_dir, "images", "*", "*.png"))
    rec_idx = (
        max(
            (int(os.path.splitext(os.path.basename(p))[0]) for p in existing),
            default=-1,
        )
        + 1
    )

    if wait:
        try:
            input("[sweep] READY -- press ENTER at the countdown...")
        except EOFError:
            pass
    sim.arm()
    time.sleep(0.2)
    s0 = sim.snapshot()
    wps = plan_sweep(gate_map, s0.pos_ned)
    print(
        f"[sweep] {len(wps)} waypoints over {len(gate_map)} gates; "
        f"recording from idx {rec_idx}",
        flush=True,
    )

    wp_i = 0
    wp_start = time.time()
    saved, rec_fid = 0, -1
    t0 = time.time()
    while time.time() - t0 < seconds and wp_i < len(wps):
        snap = sim.snapshot()
        if not snap.has_pose():
            time.sleep(1 / HZ)
            continue
        p = np.asarray(snap.pos_ned, float)
        v = np.asarray(snap.vel_ned, float)
        roll, pitch, yaw = rpy(snap.quat)
        z, vz = p[2], v[2]

        # --- record frame + range-gated label ---
        if snap.frame is not None:
            fr = sim.data.get("frame")
            if fr and fr["frame_id"] != rec_fid:
                rec_fid = fr["frame_id"]
                labels = project_gate_keypoints(
                    snap.frame.shape[:2],
                    p,
                    np.asarray(snap.quat, float),
                    gate_map,
                    max_range_m=LABEL_MAX_RANGE,
                )
                if labels or (saved % 10) < 3:  # keep ~30% negatives
                    sp = "val" if (rec_idx % 6 == 0) else "train"
                    cv2.imwrite(
                        os.path.join(out_dir, "images", sp, f"{rec_idx:06d}.png"),
                        snap.frame,
                    )
                    with open(
                        os.path.join(out_dir, "labels", sp, f"{rec_idx:06d}.txt"), "w"
                    ) as f:
                        f.write("\n".join(_yolo_pose_line(b) for b in labels))
                    rec_idx += 1
                    saved += 1
                    if saved % 50 == 0:
                        print(
                            f"[sweep] recorded {saved} (wp {wp_i}/{len(wps)})",
                            flush=True,
                        )

        # --- fly to current waypoint (fly2 law, yaw toward motion) ---
        g = wps[wp_i]
        dx, dy = g[0] - p[0], g[1] - p[1]
        dist = math.hypot(dx, dy)
        if dist < ARRIVE_R or (time.time() - wp_start) > WP_TIMEOUT_S:
            wp_i += 1
            wp_start = time.time()
            continue
        yaw_err = wrap(math.atan2(dy, dx) - yaw)
        speed = float(np.linalg.norm(v[:2]))
        align = max(0.0, 1.0 - abs(yaw_err) / 0.4)
        v_des = 2.5 * align * min(1.0, 0.3 + dist / 10.0)
        lean = float(np.clip(0.05 * (v_des - speed), -0.05, 0.12))
        e_cross = dx * math.sin(yaw) - dy * math.cos(yaw)
        tgt_roll = float(np.clip(0.04 * e_cross, -0.12, 0.12))
        tgt_z = g[2] + ZOFF

        roll_cmd = float(
            np.clip(SIGN_ROLL * K_ATT * (tgt_roll - roll), -RATE_CLIP, RATE_CLIP)
        )
        pitch_cmd = float(
            np.clip(SIGN_PITCH * K_ATT * (-lean - pitch), -RATE_CLIP, RATE_CLIP)
        )
        yaw_cmd = float(np.clip(SIGN_YAW * K_YAW * yaw_err, -YAW_CLIP, YAW_CLIP))
        thrust = float(np.clip(HOVER_T + KP_Z * (z - tgt_z) + KD_Z * vz, 0.18, 0.5))
        sim.send_attitude_rates(roll_cmd, pitch_cmd, yaw_cmd, thrust)
        time.sleep(1 / HZ)

    write_pose_data_yaml(out_dir)
    sim.send_attitude_rates(0, 0, 0, HOVER_T)
    print(
        f"[sweep] done: recorded {saved} frames, reached wp {wp_i}/{len(wps)}",
        flush=True,
    )
    os._exit(0)


def _selftest():
    gate_map = [
        {"pos": [8.0, 0.0, -2.9], "quat": [0.7071, 0, 0, 0.7071], "w": 2.72, "h": 2.72},
        {
            "pos": [14.0, 1.5, -4.0],
            "quat": [0.7071, 0, 0, 0.7071],
            "w": 2.72,
            "h": 2.72,
        },
    ]
    wps = plan_sweep(gate_map, [0.0, 0.0, 0.0])
    assert len(wps) == len(gate_map) * len(SWEEP_RANGES), len(wps)
    gc0 = np.array([8.0, 0.0, -2.9])
    # First waypoint is the farthest approach (SWEEP_RANGES[0]) to gate 0
    # (lat=0, alt=0 at k=0, gi=0).
    d0 = np.linalg.norm((wps[0] - gc0)[:2])
    assert abs(d0 - SWEEP_RANGES[0]) < 0.5, (
        f"first wp ~{SWEEP_RANGES[0]}m, got {d0:.1f}"
    )
    # Approach side: gate 0 sits at +X of start, so its approach point is nearer
    # the start (smaller x) than the gate center.
    assert wps[0][0] < gc0[0], "approach waypoint should be on the start side"
    # The 'past' waypoint (last per-gate, r<0) is beyond the gate (larger x).
    assert wps[len(SWEEP_RANGES) - 1][0] > gc0[0], "last per-gate wp past the gate"
    assert all(np.all(np.isfinite(w)) for w in wps)
    print(f"[selftest] plan_sweep OK — {len(wps)} waypoints, ranges={SWEEP_RANGES}")
    print("[selftest] OK")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=180.0)
    ap.add_argument("--no-wait", dest="wait", action="store_false")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        _selftest()
    else:
        collect(seconds=args.seconds, wait=args.wait)
