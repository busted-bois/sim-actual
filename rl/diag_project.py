"""Validate gate projection against a REAL camera frame.

Grabs one synchronized (frame, odometry) from the live sim, projects every
gate in gate_map.json into the image using rl.spec, and overlays the quad +
center. If the green quad lands on the visible orange gate, our intrinsics +
frame conventions + gate orientation are correct. Saves an overlay PNG to
inspect.

    uv run -m rl.diag_project
"""

import json
import os
import sys

import cv2
import numpy as np

from rl import spec
from rl.sim_interface import GATE_MAP_PATH, SimInterface

OUT = os.path.join(os.path.dirname(__file__), "data", "proj_check.png")
RAW = os.path.join(os.path.dirname(__file__), "data", "proj_raw.png")


def corners_world(gate):
    """Gate corners using the gate's OWN reported size (not a constant)."""
    hw, hh = gate["w"] / 2.0, gate["h"] / 2.0
    local = np.array([[0, -hw, -hh], [0, hw, -hh], [0, hw, hh], [0, -hw, hh]], float)
    R = spec.quat_to_R(np.asarray(gate["quat"], float))
    return np.asarray(gate["pos"], float) + local @ R.T


def main():
    gate_map = json.load(open(GATE_MAP_PATH))["gates"]
    sim = SimInterface()
    if not sim.wait_for_telemetry():
        print("no telemetry", flush=True)
        os._exit(1)
    snap = sim.snapshot()
    img = snap.frame.copy()
    dpos = np.asarray(snap.pos_ned, float)
    dq = np.asarray(snap.quat, float)
    print(f"drone pos={np.round(dpos, 2)} quat={np.round(dq, 3)}", flush=True)
    cv2.imwrite(RAW, snap.frame)

    for g in gate_map:
        cw = corners_world(g)
        px, in_front = spec.project(cw, dpos, dq)
        ctr, cf = spec.project(np.asarray(g["pos"], float)[None], dpos, dq)
        rng = float(np.linalg.norm(np.asarray(g["pos"], float) - dpos))
        status = "FRONT" if in_front.all() else "behind/partial"
        print(
            f"  gate {g['id']}: range={rng:5.1f}m center_px="
            f"({ctr[0, 0]:7.1f},{ctr[0, 1]:7.1f}) {status}",
            flush=True,
        )
        if in_front.all():
            poly = np.round(px).astype(np.int32).reshape(-1, 1, 2)
            cv2.polylines(img, [poly], True, (0, 255, 0), 2)
            c = (int(ctr[0, 0]), int(ctr[0, 1]))
            if 0 <= c[0] < spec.IMG_W and 0 <= c[1] < spec.IMG_H:
                cv2.circle(img, c, 4, (0, 0, 255), -1)
                cv2.putText(
                    img,
                    str(g["id"]),
                    (c[0] + 5, c[1]),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 255),
                    2,
                )
    cv2.imwrite(OUT, img)
    print(f"saved overlay -> {OUT}", flush=True)
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
