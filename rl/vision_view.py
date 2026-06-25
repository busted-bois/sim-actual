"""Standalone live vision viewer — watch the YOLO gate detector, no flight.

Connects to the sim's camera stream, runs the YOLO-pose detector + PnP on every
frame, and shows an OpenCV window with the bounding box, the 4 labeled corners
(green = used by PnP, red = rejected), and confidence + range — like the
reference repo's vision window. It does NOT arm or command the drone, so it's
safe to run anytime the sim/race is up (the drone just sits at spawn).

    uv run -m rl.vision_view          # live window; press q to quit
    uv run -m rl.vision_view --save out.mp4   # also record the annotated video
"""

from __future__ import annotations

import argparse
import time

import cv2
import numpy as np

from rl import perception as P
from rl import pnp
from rl.fly_vision import _draw_det
from rl.gatepose import visible_mask
from rl.sim_interface import SimInterface


def _pose_for_view(det, sim):
    """Range/pose for the label. Uses odometry for world pose if available,
    else a bare PnP solve just for range (display only)."""
    odo = sim.data.get("odometry")
    if odo is not None:
        dpos = np.array([odo["x"], odo["y"], odo["z"]])
        dq = np.array([odo["qw"], odo["qx"], odo["qy"], odo["qz"]])
        return P.gate_world_pose(det, dpos, dq)
    if int(visible_mask(det).sum()) >= 4:
        return pnp.estimate_pose(np.asarray(det.corners, float))
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--save", default=None, help="optional path to record annotated mp4"
    )
    args = ap.parse_args()

    sim = SimInterface()
    print("[view] waiting for camera frames... (start the race if idle)", flush=True)
    # We only need camera frames here, not odometry — poll for the first frame.
    t0 = time.monotonic()
    while sim.data.get("frame") is None and time.monotonic() - t0 < 30.0:
        time.sleep(0.05)
    if sim.data.get("frame") is None:
        print("[view] no camera frames — is the sim running?", flush=True)
        return

    perc = P.GatePerception()
    print("[view] vision up — press q in the window to quit", flush=True)

    writer = None
    last_fid = -1
    fps = 0.0
    while True:
        frame = sim.data.get("frame")
        if not frame or frame["frame_id"] == last_fid:
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            time.sleep(0.003)
            continue
        last_fid = frame["frame_id"]
        img = frame["img"]
        t = time.monotonic()
        dets = perc.infer.detect(img)
        infer_ms = (time.monotonic() - t) * 1e3
        fps = 0.9 * fps + 0.1 * (1000.0 / max(infer_ms, 1.0))

        ann = img.copy()
        n_pose = 0
        for d in dets:
            pose = _pose_for_view(d, sim)
            if pose is not None:
                n_pose += 1
            _draw_det(ann, d, pose)
        cv2.putText(
            ann,
            f"det={len(dets)} pose={n_pose} yolo={infer_ms:.0f}ms ~{fps:.0f}fps",
            (6, 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )

        if args.save:
            if writer is None:
                h, w = ann.shape[:2]
                writer = cv2.VideoWriter(
                    args.save, cv2.VideoWriter_fourcc(*"mp4v"), 20.0, (w, h)
                )
            writer.write(ann)

        cv2.imshow("vision", ann)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()
    print("[view] done", flush=True)


if __name__ == "__main__":
    main()
