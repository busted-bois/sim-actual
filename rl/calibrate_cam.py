"""Calibrate camera focal length + tilt against the real gate.

Detects the nearest gate (HSV) in a live frame, and with the known gate world
position + drone pose, backs out the focal length (from apparent size) and the
camera tilt (from vertical position). Reports recommended rl.spec values and
saves a corrected-projection overlay to verify.

    uv run -m rl.calibrate_cam
"""

import json
import os
import sys

import cv2
import numpy as np

from rl import spec
from rl.sim_interface import GATE_MAP_PATH, SimInterface
from simulator.gate_detector import detect_gate


def project_with(f, cx, cy, tilt_deg, p_world, dpos, dq):
    R_wb = spec.quat_to_R(dq)
    base = np.array([[0, 0, 1.0], [1, 0, 0], [0, 1, 0]])
    R_body_cam = spec.R_y(np.radians(tilt_deg)) @ base
    rel = np.atleast_2d(p_world) - dpos
    p_cam = (rel @ R_wb) @ R_body_cam
    z = p_cam[:, 2]
    u = f * p_cam[:, 0] / z + cx
    v = f * p_cam[:, 1] / z + cy
    return np.stack([u, v], 1), z


def main():
    gate_map = json.load(open(GATE_MAP_PATH))["gates"]
    sim = SimInterface()
    if not sim.wait_for_telemetry():
        print("no telemetry", flush=True)
        os._exit(1)
    snap = sim.snapshot()
    img = snap.frame
    dpos = np.asarray(snap.pos_ned, float)
    dq = np.asarray(snap.quat, float)
    det = detect_gate(img, 0, 0)
    if det is None:
        print(
            "HSV detector found no gate; cannot calibrate from this frame", flush=True
        )
        os._exit(1)

    u0, v0 = det.centroid_x_px, det.centroid_y_px
    w0, h0 = det.width_px, det.height_px
    # Nearest gate = gate 0 (assume drone at start facing the course).
    g0 = min(gate_map, key=lambda g: np.linalg.norm(np.asarray(g["pos"], float) - dpos))
    rng = float(np.linalg.norm(np.asarray(g0["pos"], float) - dpos))
    size = 0.5 * (g0["w"] + g0["h"])
    print(f"drone pos={np.round(dpos, 2)} quat={np.round(dq, 3)}", flush=True)
    print(
        f"detected gate {g0['id']}: center=({u0:.0f},{v0:.0f}) size=({w0:.0f}x{h0:.0f})px "
        f"range={rng:.1f}m gate_size={size:.2f}m",
        flush=True,
    )

    # Focal length from apparent size: f = px_size * range / metric_size.
    f_w = w0 * rng / size
    f_h = h0 * rng / size
    f_est = 0.5 * (f_w + f_h)
    print(
        f"focal estimate: f_w={f_w:.0f} f_h={f_h:.0f} -> f≈{f_est:.0f} "
        f"(prompt said 320)",
        flush=True,
    )

    # Tilt: scan for the tilt that reprojects gate0 center to detected v0.
    cx, cy = spec.CX, spec.CY
    best = min(
        np.arange(-40, 41, 0.25),
        key=lambda t: abs(
            project_with(
                f_est, cx, cy, t, np.asarray(g0["pos"], float)[None], dpos, dq
            )[0][0, 1]
            - v0
        ),
    )
    pu = project_with(
        f_est, cx, cy, best, np.asarray(g0["pos"], float)[None], dpos, dq
    )[0][0]
    print(
        f"tilt estimate: {best:.2f} deg (prompt said 20) -> reproj center="
        f"({pu[0]:.0f},{pu[1]:.0f}) vs detected ({u0:.0f},{v0:.0f})",
        flush=True,
    )

    # Overlay corrected projection for all gates.
    over = img.copy()
    cv2.circle(over, (int(u0), int(v0)), 6, (255, 0, 255), 2)  # detected (magenta)
    for g in gate_map:
        hw, hh = g["w"] / 2, g["h"] / 2
        local = np.array(
            [[0, -hw, -hh], [0, hw, -hh], [0, hw, hh], [0, -hw, hh]], float
        )
        cw = (
            np.asarray(g["pos"], float)
            + local @ spec.quat_to_R(np.asarray(g["quat"], float)).T
        )
        px, z = project_with(f_est, cx, cy, best, cw, dpos, dq)
        if (z > 0).all():
            cv2.polylines(
                over,
                [np.round(px).astype(np.int32).reshape(-1, 1, 2)],
                True,
                (0, 255, 0),
                2,
            )
    out = os.path.join(os.path.dirname(__file__), "data", "calib_check.png")
    cv2.imwrite(out, over)
    print(
        f"saved corrected overlay (green=calibrated proj, magenta=detected) -> {out}",
        flush=True,
    )
    print(f"\nRECOMMEND: FX=FY={f_est:.0f}, CAM_TILT_DEG={best:.1f}", flush=True)
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
