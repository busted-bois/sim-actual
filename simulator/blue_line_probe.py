"""Passive A/B probe for the blue-line estimators — no MAVLink, no arming.

Live mode reads ONLY the camera stream (UDP 5600) and scores the inner-edge
estimator against the legacy centroid estimator on identical frames. Safe to
run during a live race; it never commands the drone.

    uv run -m simulator.blue_line_probe              (or: make bl-probe)
    uv run -m simulator.blue_line_probe --replay runs/bl_probe/raw

With the drone PARKED the true corridor offset is constant, so every
frame-to-frame change in cx is measurement noise. The other headline number is
the FALLBACK RATE: how often the inner-edge scan failed to track at all and
silently handed back the centroid estimate.

Live mode saves raw frames to runs/bl_probe/raw so --replay can re-run the
estimator on the exact same pixels as many times as tuning needs, without a
race running. Annotated overlays land in runs/bl_probe/.
"""

from __future__ import annotations

import argparse
import glob
import os
import statistics as st
import time
from collections import Counter

import cv2

_SECONDS = float(os.environ.get("BL_PROBE_SECONDS", "20"))
_WAIT_S = float(os.environ.get("BL_PROBE_WAIT", "600"))  # idle wait for a race
_SAVE_EVERY = int(os.environ.get("BL_PROBE_SAVE_EVERY", "30"))
_OUT_DIR = os.path.join("runs", "bl_probe")
_RAW_DIR = os.path.join(_OUT_DIR, "raw")


def _jitter(vals: list[float]) -> float:
    """Mean |frame-to-frame| change — pure noise when the drone is parked."""
    if len(vals) < 2:
        return float("nan")
    return st.mean(abs(b - a) for a, b in zip(vals[:-1], vals[1:]))


class _Stats:
    """Accumulates per-frame estimates from either live or replay input."""

    # Score the estimators themselves, never the flown value: in shadow mode
    # cx_norm IS the centroid, so comparing it would report zero disagreement.
    _SERIES = (
        ("cx  edge", "cx_edge"),
        ("cx  cent", "cx_centroid"),
        ("hdg edge", "heading_edge"),
        ("hdg cent", "heading_centroid"),
    )

    def __init__(self):
        self.series: dict[str, list[float]] = {k: [] for _, k in self._SERIES}
        self.rows: list[int] = []
        self.src: Counter[str] = Counter()
        self.n = 0
        self.found = 0

    def add(self, bl: dict) -> None:
        self.n += 1
        if not bl.get("found"):
            return
        self.found += 1
        self.src[str(bl.get("source", "?"))] += 1
        self.rows.append(int(bl["edge_rows"]))
        vals = [float(bl.get(key, float("nan"))) for _, key in self._SERIES]
        if any(v != v for v in vals):
            return  # edge not computed on this frame — nothing to compare
        for (_, key), v in zip(self._SERIES, vals):
            self.series[key].append(v)

    def report(self) -> None:
        print("\n=== blue-line estimator A/B ===", flush=True)
        print(
            f"frames={self.n} found={self.found} "
            f"({100.0 * self.found / max(self.n, 1):.0f}%)",
            flush=True,
        )
        if not self.found:
            print("no corridor detected — point the drone down the track", flush=True)
            return
        # Headline: how often did the edge scan actually track? A high fallback
        # share means the scan is quitting, not that it is estimating badly.
        tracked = self.src.get("edge", 0) + self.src.get("shadow", 0)
        print(
            f"source: {dict(self.src)} — edge tracked {tracked}/{self.found} "
            f"({100.0 * tracked / self.found:.0f}%), "
            f"fell back {self.src.get('centroid', 0)} "
            f"({100.0 * self.src.get('centroid', 0) / self.found:.0f}%)",
            flush=True,
        )
        if not self.series["cx_edge"]:
            print("edge estimator never ran — BL_INNER_EDGE=0?", flush=True)
            return
        tracked = [r for r in self.rows if r > 0]
        if tracked:
            print(
                f"edge rows when engaged: mean={st.mean(tracked):.1f} "
                f"min={min(tracked)} max={max(tracked)}",
                flush=True,
            )
        for label, key in self._SERIES:
            vals = self.series[key]
            print(
                f"{label}: mean={st.mean(vals):+.4f} sd={st.pstdev(vals):.4f} "
                f"jitter={_jitter(vals):.4f}",
                flush=True,
            )
        edge, cent = self.series["cx_edge"], self.series["cx_centroid"]
        disagree = [abs(a - b) for a, b in zip(edge, cent)]
        print(
            f"|cx_edge - cx_cent|: mean={st.mean(disagree):.4f} "
            f"max={max(disagree):.4f}",
            flush=True,
        )
        je, jc = _jitter(edge), _jitter(cent)
        if len(edge) >= 2 and jc > 1e-9:
            verdict = "QUIETER" if je < jc else "NOISIER"
            print(
                f"\n-> inner-edge cx is {verdict}: {je / jc:.2f}x centroid jitter",
                flush=True,
            )


def _replay(src_dir: str) -> None:
    """Re-run the estimator over saved raw frames — no sim needed."""
    from simulator.blue_line_vision import (
        annotate_blue_lines,
        detect_blue_lines,
        estimate_to_dict,
    )

    paths = sorted(glob.glob(os.path.join(src_dir, "*.png")))
    if not paths:
        print(f"[bl-probe] no frames in {src_dir}", flush=True)
        return
    out_dir = os.path.join(src_dir, "annotated")
    os.makedirs(out_dir, exist_ok=True)
    print(f"[bl-probe] replaying {len(paths)} frames from {src_dir}", flush=True)

    stats = _Stats()
    for i, path in enumerate(paths):
        img = cv2.imread(path)
        if img is None:
            continue
        est, mask = detect_blue_lines(img, i, return_mask=True)
        stats.add(estimate_to_dict(est))
        cv2.imwrite(
            os.path.join(out_dir, os.path.basename(path)[:-4] + ".jpg"),
            annotate_blue_lines(img, est, mask),
        )
    stats.report()
    print(f"overlays -> {out_dir}/", flush=True)


def _live() -> None:
    os.environ.setdefault("SKIP_YOLO", "1")  # probe only needs the HSV path
    from simulator.vision_rx import VisionRX

    data: dict = {}
    VisionRX(data)
    os.makedirs(_RAW_DIR, exist_ok=True)
    print(f"[bl-probe] listening {_SECONDS:.0f}s — frames -> {_OUT_DIR}/", flush=True)

    # The sim only streams while a race is running, so start the capture clock
    # on the FIRST frame rather than on wall time — otherwise the window
    # silently expires before the race is started and nothing is captured.
    deadline = time.monotonic() + _WAIT_S
    while data.get("blue_line") is None and time.monotonic() < deadline:
        time.sleep(0.05)
    if data.get("blue_line") is None:
        print(
            f"[bl-probe] no frames in {_WAIT_S:.0f}s — is a race running?", flush=True
        )
        return
    print(f"[bl-probe] frames arriving — capturing {_SECONDS:.0f}s", flush=True)

    stats = _Stats()
    saved = 0
    last_fid = None
    t_end = time.monotonic() + _SECONDS
    try:
        while time.monotonic() < t_end:
            # Key off blue_line's OWN frame_id: vision_rx swaps in data["frame"]
            # before it has attached the estimate or the annotation.
            bl = data.get("blue_line")
            if bl is None or bl["frame_id"] == last_fid:
                time.sleep(0.002)
                continue
            last_fid = bl["frame_id"]
            stats.add(bl)
            frame = data.get("frame")
            if (
                frame is not None
                and frame["frame_id"] == last_fid
                and stats.n % _SAVE_EVERY == 0
            ):
                # Raw frame stays lossless PNG — it is the --replay input, so the
                # estimator can be re-run on the exact same pixels. The annotated
                # copy is only for eyeballing, and PNG-encoding it here blinds the
                # poll loop for ~26 ms, so it goes out as JPEG.
                saved += 1
                cv2.imwrite(
                    os.path.join(_RAW_DIR, f"f{last_fid:06d}.png"), frame["img"]
                )
                ann = frame.get("annotated")
                if ann is not None:
                    cv2.imwrite(os.path.join(_OUT_DIR, f"f{last_fid:06d}.jpg"), ann)
            if stats.n % 60 == 0:
                print(
                    f"[bl-probe] n={stats.n} src={bl.get('source')} "
                    f"cx_edge={bl.get('cx_norm', 0):+.3f} "
                    f"cx_cent={bl.get('cx_centroid', 0):+.3f} "
                    f"rows={bl.get('edge_rows')}",
                    flush=True,
                )
    except KeyboardInterrupt:
        pass
    stats.report()
    print(f"saved {saved} raw frames to {_RAW_DIR}/", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--replay",
        metavar="DIR",
        help="score saved raw frames instead of the live camera stream",
    )
    args = ap.parse_args()
    if args.replay:
        _replay(args.replay)
        return
    _live()
    os._exit(0)  # hard-exit past the non-daemon VisionRX receiver thread


if __name__ == "__main__":
    main()
