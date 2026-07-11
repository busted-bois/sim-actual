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
    gate_map = [
        {
            "id": 0,
            "pos": [6.0, 0.0, 0.0],
            "quat": [0.0, 0.0, 0.0, 1.0],
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
    print("[selftest] OK — labeling projects + culls correctly")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=1200)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        _selftest()
    else:
        collect(args.frames)
