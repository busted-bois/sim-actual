"""Module 2 — Dataset generation (frames + auto-labeled segmentation masks).

Auto-labeling: with known camera intrinsics (rl.spec), the gate map, and the
drone pose at capture time, every gate's 4 corners project deterministically
into the image. Filling those quads yields a pixel-perfect gate mask — no
manual labeling.

Data collection flies the existing vision pilot (velocity setpoints don't
actuate this sim, so we reuse the proven attitude-rate pilot to produce
diverse gate approaches) and records frames + masks at camera rate.

    uv run -m rl.dataset            # collect from live sim
    uv run -m rl.dataset --frames 1500
"""

from __future__ import annotations

import argparse
import json
import os
import time

import cv2
import numpy as np

from rl import spec

OUT_DIR = os.path.join(os.path.dirname(__file__), "data", "gatenet_ds")
POSE_OUT_DIR = os.path.join(os.path.dirname(__file__), "data", "gate_pose_ds")


# ----------------------------------------------------------------------------
# Auto-labeling: project gate geometry -> binary mask
# ----------------------------------------------------------------------------
def project_gate_mask(
    frame_hw: tuple[int, int],
    drone_pos: np.ndarray,
    drone_quat: np.ndarray,
    gate_map: list,
    min_corners_front: int = 4,
) -> np.ndarray:
    """Return a uint8 {0,255} mask of all gate openings visible from this pose.

    A gate is drawn only if all 4 corners are in front of the camera (avoids
    wild projections when a gate straddles the image plane).
    """
    h, w = frame_hw
    mask = np.zeros((h, w), dtype=np.uint8)
    dpos = np.asarray(drone_pos, dtype=np.float64)
    dq = np.asarray(drone_quat, dtype=np.float64)

    for g in gate_map:
        gpos = np.asarray(g["pos"], dtype=np.float64)
        gquat = np.asarray(g["quat"], dtype=np.float64)
        corners = spec.gate_corners_world(gpos, gquat)
        px, in_front = spec.project(corners, dpos, dq)
        if int(in_front.sum()) < min_corners_front:
            continue
        # Reject gates entirely off-screen (cheap bbox test).
        if px[:, 0].max() < 0 or px[:, 0].min() > w:
            continue
        if px[:, 1].max() < 0 or px[:, 1].min() > h:
            continue
        poly = np.round(px).astype(np.int32).reshape(-1, 1, 2)
        cv2.fillPoly(mask, [poly], 255)
    return mask


# ----------------------------------------------------------------------------
# Auto-labeling: project gate corners -> YOLO-pose keypoint labels
# ----------------------------------------------------------------------------
def project_gate_keypoints(
    frame_hw: tuple[int, int],
    drone_pos: np.ndarray,
    drone_quat: np.ndarray,
    gate_map: list,
    min_corners_onscreen: int = 2,
    max_range_m: float | None = None,
) -> list[dict]:
    """Per-visible-gate YOLO-pose labels (normalized to [0,1]).

    Each entry: {"kpts": [(xn,yn,v)*4], "bbox": (cxn,cyn,wn,hn)} in TL,TR,BR,BL
    order. Off-screen / behind-camera corners get v=0 with zeroed coords (the
    clip-to-edge fix: never teach the model to predict a corner at the border).
    The bbox spans only on-screen corners. A gate is emitted only if at least
    `min_corners_onscreen` corners are on-screen (else the bbox is degenerate).

    max_range_m: if set, gates farther than this (drone->gate distance) are not
    labeled — they're tiny specks the detector can't learn and they only drag
    recall down (range-gating, fixes the far-bias).
    """
    h, w = frame_hw
    dpos = np.asarray(drone_pos, dtype=np.float64)
    dq = np.asarray(drone_quat, dtype=np.float64)
    out: list[dict] = []
    for g in gate_map:
        if max_range_m is not None and (
            np.linalg.norm(np.asarray(g["pos"], dtype=np.float64) - dpos) > max_range_m
        ):
            continue
        px, in_front = spec.gate_keypoints_px(g["pos"], g["quat"], dpos, dq)
        kpts, xs, ys = [], [], []
        for i in range(4):
            x, y = float(px[i, 0]), float(px[i, 1])
            onscreen = bool(in_front[i]) and 0.0 <= x < w and 0.0 <= y < h
            if onscreen:
                kpts.append((x / w, y / h, 2))
                xs.append(x)
                ys.append(y)
            else:
                kpts.append((0.0, 0.0, 0))
        if len(xs) < min_corners_onscreen:
            continue
        x0, x1 = max(0.0, min(xs)), min(float(w), max(xs))
        y0, y1 = max(0.0, min(ys)), min(float(h), max(ys))
        bw, bh = (x1 - x0) / w, (y1 - y0) / h
        if bw <= 0 or bh <= 0:
            continue
        cx, cy = (x0 + x1) / 2 / w, (y0 + y1) / 2 / h
        out.append({"kpts": kpts, "bbox": (cx, cy, bw, bh)})
    return out


def _yolo_pose_line(lbl: dict) -> str:
    """One Ultralytics pose label row: `cls cx cy w h  x1 y1 v1 ... x4 y4 v4`."""
    cx, cy, bw, bh = lbl["bbox"]
    parts: list = [0, cx, cy, bw, bh]
    for x, y, v in lbl["kpts"]:
        parts += [x, y, v]
    return " ".join(str(p) if isinstance(p, int) else f"{p:.6f}" for p in parts)


def write_pose_data_yaml(out_dir: str = POSE_OUT_DIR) -> str:
    """Write the Ultralytics dataset config; returns its path."""
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "data.yaml")
    with open(path, "w") as f:
        f.write(
            f"# Auto-generated by rl.dataset — YOLO11-pose gate-corner labels\n"
            f"path: {os.path.abspath(out_dir)}\n"
            "train: images/train\n"
            "val: images/val\n"
            f"kpt_shape: [{spec.N_KEYPOINTS}, 3]\n"
            f"flip_idx: {spec.KPT_FLIP_IDX}\n"
            "names:\n  0: gate\n"
        )
    return path


# ----------------------------------------------------------------------------
# Live collection — YOLO-pose dataset (images + keypoint labels)
# ----------------------------------------------------------------------------
def collect_pose(n_frames: int = 1500, out_dir: str = POSE_OUT_DIR, val_every: int = 6):
    """Fly the proven attitude-rate pilot, record frames + auto-keypoint labels
    in Ultralytics layout (images/{train,val}, labels/{train,val}, data.yaml).
    Empty label files are kept as negative samples (false-positive suppression).
    """
    from simulator.controller import Controller

    from rl.sim_interface import GATE_MAP_PATH, SimInterface, load_gate_map

    for split in ("train", "val"):
        os.makedirs(os.path.join(out_dir, "images", split), exist_ok=True)
        os.makedirs(os.path.join(out_dir, "labels", split), exist_ok=True)

    sim = SimInterface()
    if not sim.wait_for_telemetry():
        print("[ds-pose] no telemetry — is the simulator running?")
        return
    # The gate map is broadcast once at race start; if we joined mid-race and
    # missed it, fall back to the saved map (the course/origin is static, so a
    # previously-captured gate_map.json projects correctly for auto-labeling).
    gate_map = sim.capture_gate_map(timeout_s=5.0)
    if not gate_map and os.path.exists(GATE_MAP_PATH):
        gate_map = load_gate_map(GATE_MAP_PATH)
        print(f"[ds-pose] using saved gate map ({len(gate_map)} gates)", flush=True)
    if not gate_map:
        print("[ds-pose] no gate map (no broadcast, no saved file); aborting")
        return

    controller = Controller(sim.conn, sim.data, sim.system_boot_ms)
    sim.arm()
    print("[ds-pose] armed; recording...", flush=True)

    saved = 0
    last_frame_id = -1
    t_last_reset = time.monotonic()
    meta = []  # per-frame drone pose, so labels can be re-projected/debugged offline
    try:
        while saved < n_frames:
            controller.update()
            frame = sim.data.get("frame")
            odo = sim.data.get("odometry")
            if not frame or odo is None or frame["frame_id"] == last_frame_id:
                continue
            last_frame_id = frame["frame_id"]

            img = frame["img"]
            dpos = np.array([odo["x"], odo["y"], odo["z"]])
            dq = np.array([odo["qw"], odo["qx"], odo["qy"], odo["qz"]])
            labels = project_gate_keypoints(img.shape[:2], dpos, dq, gate_map)

            # Balanced set: keep ~30% of gate-less frames as negatives.
            if not labels and (saved % 10) >= 3:
                continue

            split = "val" if (saved % val_every == 0) else "train"
            cv2.imwrite(os.path.join(out_dir, "images", split, f"{saved:06d}.png"), img)
            with open(
                os.path.join(out_dir, "labels", split, f"{saved:06d}.txt"), "w"
            ) as f:
                f.write("\n".join(_yolo_pose_line(b) for b in labels))

            # First few labeled frames: dump an overlay so label alignment is
            # eyeball-verifiable before committing to a full collection/train.
            if labels and saved < 12:
                dbg = img.copy()
                h, w = img.shape[:2]
                for b in labels:
                    pts = np.array(
                        [(x * w, y * h) for x, y, v in b["kpts"] if v > 0], np.int32
                    )
                    for px_, py_ in pts:
                        cv2.circle(dbg, (px_, py_), 4, (0, 255, 0), -1)
                    if len(pts) >= 3:
                        cv2.polylines(dbg, [pts], True, (0, 255, 0), 2)
                os.makedirs(os.path.join(out_dir, "debug"), exist_ok=True)
                cv2.imwrite(os.path.join(out_dir, "debug", f"{saved:06d}.png"), dbg)
            meta.append(
                {
                    "idx": saved,
                    "split": split,
                    "pos": dpos.tolist(),
                    "quat": dq.tolist(),
                    "frame_time_ns": frame["sim_time_ns"],
                    "n_gates": len(labels),
                }
            )
            saved += 1
            if saved % 50 == 0:
                print(f"[ds-pose] {saved}/{n_frames} frames", flush=True)

            if time.monotonic() - t_last_reset > 45.0:
                sim.reset_sim()
                t_last_reset = time.monotonic()
    finally:
        with open(os.path.join(out_dir, "meta.json"), "w") as f:
            json.dump({"gate_map": gate_map, "frames": meta}, f, indent=2)
        write_pose_data_yaml(out_dir)
        print(f"[ds-pose] done: {saved} frames -> {out_dir}", flush=True)


# ----------------------------------------------------------------------------
# Live collection
# ----------------------------------------------------------------------------
def collect(n_frames: int = 1200, out_dir: str = OUT_DIR):
    from simulator.controller import Controller

    from rl.sim_interface import SimInterface

    img_dir = os.path.join(out_dir, "images")
    mask_dir = os.path.join(out_dir, "masks")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)

    sim = SimInterface()
    if not sim.wait_for_telemetry():
        print("[ds] no telemetry — is the simulator running?")
        return
    gate_map = sim.capture_gate_map()
    if not gate_map:
        print("[ds] no gate map; aborting")
        return

    # Drive the existing vision pilot in a background-ish loop while we record.
    controller = Controller(sim.conn, sim.data, sim.system_boot_ms)
    sim.arm()
    print("[ds] armed; recording...", flush=True)

    saved = 0
    last_frame_id = -1
    meta = []
    t_last_reset = time.monotonic()
    try:
        while saved < n_frames:
            controller.update()  # one pilot tick (~250 Hz, includes sleep)

            frame = sim.data.get("frame")
            odo = sim.data.get("odometry")
            if not frame or odo is None:
                continue
            if frame["frame_id"] == last_frame_id:
                continue
            last_frame_id = frame["frame_id"]

            img = frame["img"]
            dpos = np.array([odo["x"], odo["y"], odo["z"]])
            dq = np.array([odo["qw"], odo["qx"], odo["qy"], odo["qz"]])
            mask = project_gate_mask(img.shape[:2], dpos, dq, gate_map)

            # Keep a balanced set: skip ~70% of empty-mask frames.
            if mask.max() == 0 and (saved % 10) >= 3:
                continue

            cv2.imwrite(os.path.join(img_dir, f"{saved:06d}.png"), img)
            cv2.imwrite(os.path.join(mask_dir, f"{saved:06d}.png"), mask)
            meta.append(
                {
                    "idx": saved,
                    "pos": dpos.tolist(),
                    "quat": dq.tolist(),
                    "frame_time_ns": frame["sim_time_ns"],
                    "gate_px_visible": int(mask.max() > 0),
                }
            )
            saved += 1
            if saved % 50 == 0:
                print(f"[ds] {saved}/{n_frames} frames", flush=True)

            # Periodic reset keeps the run on-course and diversifies spawns.
            if time.monotonic() - t_last_reset > 45.0:
                sim.reset_sim()
                t_last_reset = time.monotonic()
    finally:
        with open(os.path.join(out_dir, "meta.json"), "w") as f:
            json.dump({"gate_map": gate_map, "frames": meta}, f, indent=2)
        print(f"[ds] done: {saved} frames -> {out_dir}", flush=True)


# ----------------------------------------------------------------------------
# Offline self-test (no sim): synthesize a pose looking at a gate, check mask.
# ----------------------------------------------------------------------------
def _selftest():
    # Gate centered in the 20°-up-tilted camera: z ~ -range*tan(20°) so all 4
    # corners frame on-screen (a gate at the drone's own altitude sits low/
    # partly below the frame because the camera looks up).
    gate_map = [
        {
            "id": 0,
            "pos": [8.0, 0.0, -2.9],
            "quat": [0.70710678, 0.0, 0.0, 0.70710678],  # 90°-about-Z: faces drone
            "w": 1.5,
            "h": 1.5,
        }
    ]
    dpos = np.array([0.0, 0.0, 0.0])
    dq = np.array([1.0, 0.0, 0.0, 0.0])  # facing North toward gate
    mask = project_gate_mask((spec.IMG_H, spec.IMG_W), dpos, dq, gate_map)
    ys, xs = np.where(mask > 0)
    assert len(xs) > 0, "gate should be visible"
    cx, cy = xs.mean(), ys.mean()
    frac = (mask > 0).mean()
    print(
        f"[selftest] mask pixels={len(xs)} centroid=({cx:.0f},{cy:.0f}) fill={frac:.3f}"
    )
    assert abs(cx - spec.CX) < 5, "gate ahead should center horizontally"
    # Off-screen gate -> empty mask
    empty = project_gate_mask(
        (spec.IMG_H, spec.IMG_W), dpos, np.array([0.7071, 0, 0, 0.7071]), gate_map
    )  # facing East
    assert empty.max() == 0, "gate behind/aside should give empty mask"

    # YOLO-pose keypoint labels: head-on gate -> 4 visible corners + valid bbox.
    labels = project_gate_keypoints((spec.IMG_H, spec.IMG_W), dpos, dq, gate_map)
    assert len(labels) == 1, "gate ahead should produce one label"
    kp = labels[0]["kpts"]
    assert sum(1 for *_, v in kp if v == 2) == 4, "all 4 corners on-screen + front"
    for x, y, v in kp:
        assert 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0, "normalized kpts in [0,1]"
    cx, cy, bw, bh = labels[0]["bbox"]
    assert 0 < bw <= 1 and 0 < bh <= 1, "valid normalized bbox"
    assert abs(cx - spec.CX / spec.IMG_W) < 0.05, "bbox centered on a head-on gate"
    # Round-trips through the label-line writer.
    line = _yolo_pose_line(labels[0])
    assert len(line.split()) == 5 + 3 * spec.N_KEYPOINTS, "cls+bbox+4*(x,y,v)"
    # Off-screen gate -> no keypoint label.
    empty_kp = project_gate_keypoints(
        (spec.IMG_H, spec.IMG_W), dpos, np.array([0.7071, 0, 0, 0.7071]), gate_map
    )
    assert empty_kp == [], "gate behind/aside should produce no keypoint label"

    # Range-gating: a far gate is dropped when max_range_m is set.
    far_map = gate_map + [
        {
            "id": 1,
            "pos": [30.0, 0.0, -11.0],
            "quat": [0.70710678, 0, 0, 0.70710678],
            "w": 2.72,
            "h": 2.72,
        }
    ]
    full = project_gate_keypoints((spec.IMG_H, spec.IMG_W), dpos, dq, far_map)
    gated = project_gate_keypoints(
        (spec.IMG_H, spec.IMG_W), dpos, dq, far_map, max_range_m=15.0
    )
    assert len(full) == 2, f"both gates should label without gating, got {len(full)}"
    assert len(gated) == 1, f"far gate should be range-gated out, got {len(gated)}"
    print("[selftest] OK — labeling projects + culls + range-gates (mask + keypoints)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=1200)
    ap.add_argument(
        "--pose",
        action="store_true",
        help="collect a YOLO11-pose keypoint dataset (vs the U-Net mask dataset)",
    )
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        _selftest()
    elif args.pose:
        collect_pose(args.frames)
    else:
        collect(args.frames)
