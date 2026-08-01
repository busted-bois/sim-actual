"""Pull frames out of a run recording, by time.

Reviewing a 65 MB mp4 to check one moment means playing it. This writes the
frames you actually want to disk instead, so a specific instant can be looked
at directly -- pair it with the run's telemetry CSV to go from "vD spikes at
t=41.2" to the picture at t=41.2.

    uv run scripts/video_frames.py <mp4> --at 12.5
    uv run scripts/video_frames.py <mp4> --every 5 --out runs/archive/frames/x

Normally reached through `make videos-frames RUN=<name> AT=12.5`.

Two things make naive seeking wrong here, both handled below:

* The recordings are mp4v (MPEG-4 Part 2) with P-frames, so seeking by
  CAP_PROP_POS_MSEC lands on the nearest keyframe -- off by up to a second.
  We seek by frame index and step the remainder.
* The container header says 30 fps, but display.py only writes a frame when the
  camera delivers one, so the real rate is lower and varies per run. The run's
  sidecar records the measured rate as video.fps_actual; pass it with --fps.
  Without it the timings here are the container's, and will drift.

This is a leaf tool: invoked from a Make target, never imported by flight code.
"""

import argparse
import json
import os
import sys

import cv2


def _fps_from_sidecar(video_path):
    """Prefer the measured rate the run recorded next to its mp4."""
    for cand in (
        os.path.splitext(video_path)[0] + ".json",
        # Published name keeps the member prefix the local one lacks.
        os.path.join(os.path.dirname(video_path), os.path.basename(video_path)[:-4] + ".json"),
    ):
        try:
            with open(cand, encoding="utf-8") as fh:
                fps = (json.load(fh).get("video") or {}).get("fps_actual")
            if fps:
                return float(fps), cand
        except (OSError, ValueError, AttributeError):
            continue
    return None, None


def grab(cap, index):
    """Read frame `index` exactly. Seeking alone lands on a keyframe, so step
    forward from wherever the seek actually put us."""
    cap.set(cv2.CAP_PROP_POS_FRAMES, index)
    landed = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
    for _ in range(max(0, index - landed)):
        if not cap.grab():
            return None
    ok, frame = cap.read()
    return frame if ok else None


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("video")
    ap.add_argument("--at", type=float, action="append", default=[],
                    help="second to extract; repeatable")
    ap.add_argument("--every", type=float, help="extract one frame every N seconds")
    ap.add_argument("--fps", type=float,
                    help="true frame rate; defaults to the sidecar's video.fps_actual")
    ap.add_argument("--out", help="output directory (default: alongside the mp4)")
    args = ap.parse_args(argv)

    if not os.path.exists(args.video):
        print(f"no such video: {args.video}", file=sys.stderr)
        return 1

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"could not open: {args.video}", file=sys.stderr)
        return 1
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fps, src = args.fps, "--fps"
    if fps is None:
        fps, sidecar = _fps_from_sidecar(args.video)
        src = f"sidecar {os.path.basename(sidecar)}" if sidecar else None
    if fps is None:
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        src = "container header (no sidecar -- timings may drift)"
    duration = total / fps if fps else 0.0
    print(f"{total} frames @ {fps:.2f} fps ({src}) = {duration:.1f}s")

    times = list(args.at)
    if args.every:
        t = 0.0
        while t < duration:
            times.append(round(t, 3))
            t += args.every
    if not times:
        print("nothing to do: pass --at or --every", file=sys.stderr)
        return 2

    out = args.out or os.path.join(
        os.path.dirname(args.video), os.path.splitext(os.path.basename(args.video))[0]
    )
    os.makedirs(out, exist_ok=True)

    written = 0
    for t in times:
        index = int(round(t * fps))
        if not 0 <= index < max(total, 1):
            print(f"  t={t:.2f}s outside the recording, skipped")
            continue
        frame = grab(cap, index)
        if frame is None:
            print(f"  t={t:.2f}s could not be read, skipped")
            continue
        path = os.path.join(out, f"t{t:08.2f}.jpg")
        cv2.imwrite(path, frame)
        print(f"  t={t:7.2f}s -> {path}")
        written += 1
    cap.release()
    print(f"{written} frame(s) -> {out}")
    return 0 if written else 3


if __name__ == "__main__":
    raise SystemExit(main())
