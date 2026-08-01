"""Offline replay of the occlusion PEEK detector over a recorded run video.

Runs the SAME vision path the live pilot uses -- detect_gate + obstacles_from_frame
+ OcclusionTracker -- over a window of frames and reports whether the cue behaves:
it must be near-silent on clean gate approaches and fire a stable, CORRECT side
where a pillar occludes the gate. This is the gate BEFORE flying: a plausible
detector change has already been flight-reverted (c0d51f9), so validate offline.

    uv run scripts/peek_replay.py runs/videos/vision_XXXX.mp4 --start 2250 --count 300
    uv run scripts/peek_replay.py <mp4> --start 900 --count 600 --json clean.json
    uv run scripts/peek_replay.py <mp4> ...  --json after.json --baseline before.json

Metrics:
    fire_rate       frac of frames a cue fired      (want ~0 on a clean window)
    side_flips_s    side changes per second         (noise the roll would feel)
    agree_rate      frac of fired frames whose side points to the gate's side
    strength_p50/p90
"""

import argparse
import json
import os
import sys

import numpy as np

# Run as a bare script (uv run scripts/peek_replay.py ...) -> put the repo root on
# the path so `import simulator` resolves.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _open(path):
    import cv2

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise SystemExit(f"cannot open video: {path}")
    return cap


def replay(path, start, count):
    import cv2

    from simulator.gate_detector import detect_gate
    from simulator.gate_occlusion import OcclusionTracker, obstacles_from_frame

    cap = _open(path)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, start))
    tr = OcclusionTracker()
    rows = []
    i = start
    while count <= 0 or i < start + count:
        ok, img = cap.read()
        if not ok:
            break
        h, w = img.shape[:2]
        det = detect_gate(img, i, 0)
        gate_desc, gate_cx = None, None
        if det is not None:
            gcx, gcy = float(det.centroid_x_px), float(det.centroid_y_px)
            gw2, gh2 = float(det.width_px) / 2.0, float(det.height_px) / 2.0
            gate_desc = {"cx": gcx, "cy": gcy, "x0": gcx - gw2, "x1": gcx + gw2,
                         "y0": gcy - gh2, "y1": gcy + gh2}
            gate_cx = gcx
        occ = tr.update(gate_desc, obstacles_from_frame(img, det), (h, w),
                        bgr=img, frame_id=i)
        rows.append({
            "frame": i, "fired": occ is not None,
            "side": int(occ["side"]) if occ else 0,
            "strength": float(occ["strength"]) if occ else 0.0,
            "blob_x": occ.get("blob_x") if occ else None,
            "gate_cx": gate_cx, "reason": occ.get("reason") if occ else "",
        })
        i += 1
    cap.release()
    return rows, fps


def measure(rows, fps):
    n = len(rows)
    fired = [r for r in rows if r["fired"]]
    flips, prev = 0, 0
    for r in rows:
        s = r["side"]
        if s != 0:
            if prev != 0 and s != prev:
                flips += 1
            prev = s
    dur = n / fps if fps else 1.0
    agree_den = [r for r in fired if r["gate_cx"] is not None and r["blob_x"] is not None]
    agree_num = sum(1 for r in agree_den
                    if r["side"] == (1 if r["gate_cx"] > r["blob_x"] else -1))
    agree_rate = (agree_num / len(agree_den)) if agree_den else None
    strengths = [r["strength"] for r in fired]
    return {
        "frames": n, "fps": round(fps, 3),
        "fire_rate": round(len(fired) / n, 3) if n else 0.0,
        "side_flips_s": round(flips / dur, 3) if dur else 0.0,
        "agree_rate": round(agree_rate, 3) if agree_rate is not None else None,
        "strength_p50": round(float(np.percentile(strengths, 50)), 3) if strengths else 0.0,
        "strength_p90": round(float(np.percentile(strengths, 90)), 3) if strengths else 0.0,
    }


_ROWS = ["frames", "fps", "fire_rate", "side_flips_s", "agree_rate",
         "strength_p50", "strength_p90"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--count", type=int, default=600)
    ap.add_argument("--json", help="write metrics json")
    ap.add_argument("--baseline", help="compare against a metrics json")
    args = ap.parse_args()

    rows, fps = replay(args.video, args.start, args.count)
    m = measure(rows, fps)
    print(f"[peek_replay] {args.video} frames {args.start}..{args.start + len(rows)}")
    for k in _ROWS:
        print(f"  {k:14s} {m[k]}")
    if args.json:
        with open(args.json, "w") as f:
            json.dump(m, f, indent=2)
        print(f"[peek_replay] wrote {args.json}")
    if args.baseline:
        with open(args.baseline) as f:
            base = json.load(f)
        print("\n  metric         baseline -> now")
        for k in _ROWS:
            print(f"  {k:14s} {base.get(k)} -> {m[k]}")


if __name__ == "__main__":
    main()
